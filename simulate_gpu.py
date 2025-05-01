#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import torch

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # model dimensions
    I: int = 2
    batch: int = 1_000
    steps: int = 1_000_000

    # discretisation sizes
    Np: int = 31
    Nv: int = 10
    Nx: int = 15
    iota: float = 0.1

    # stochastic parameters
    v_bar: float = 1.0
    sigma_v: float = 1.0
    sigma_u: float = 0.1

    # learning / discount
    rho: float = 0.95
    alpha: float = 0.01
    beta: float = 1e-5
    xi: float = 500.0

    # state-space parameters
    chi_N: float = 0.05
    chi_M: float = 0.10
    lambda_N: float = 0.12
    lambda_M: float = 0.18

    # OLS window and MM aggressiveness
    Tm: int = 10_000
    theta: float = 0.1

    # device for torch tensors
device: str = "cuda"


# -----------------------------------------------------------------------------
# 2. Discretisation helpers
# -----------------------------------------------------------------------------
def _grid_linspace(low: float, high: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a 1-D tensor of n points from low to high."""
    return torch.linspace(low, high, n, dtype=dtype, device=device)


def build_discretisations(cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build price, value, and inventory grids on the specified device."""
    device = torch.device(cfg.device)
    # ----- v grid via inverse CDF of standard normal -----
    probs = (2 * torch.arange(1, cfg.Nv + 1, device=device, dtype=torch.float64) - 1) / (2 * cfg.Nv)
    # norm_ppf(x) = sqrt(2) * erfinv(2x - 1)
    v_disc = cfg.v_bar + cfg.sigma_v * torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(2 * probs - 1)

    # ----- x grid -----
    span_x = abs(cfg.chi_N - cfg.chi_M)
    low_x = -max(cfg.chi_N, cfg.chi_M) - cfg.iota * span_x
    high_x = max(cfg.chi_N, cfg.chi_M) + cfg.iota * span_x
    x_disc = _grid_linspace(low_x, high_x, cfg.Nx, device, torch.float64)

    # ----- p grid -----
    lam_max = max(cfg.lambda_N, cfg.lambda_M)
    ph = cfg.v_bar + lam_max * (cfg.I * max(cfg.chi_N, cfg.chi_M) + cfg.sigma_u * 1.96)
    pl = cfg.v_bar - lam_max * (cfg.I * max(cfg.chi_N, cfg.chi_M) + cfg.sigma_u * 1.96)
    span_p = ph - pl
    p_disc = _grid_linspace(pl - cfg.iota * span_p, ph + cfg.iota * span_p, cfg.Np, device, torch.float64)

    return p_disc, v_disc, x_disc


# -----------------------------------------------------------------------------
# 3. Ring buffer & Online OLS
# -----------------------------------------------------------------------------
class Ring:
    """Circular buffer storing a fixed number of scalars on a torch device."""
    __slots__ = ("buf", "idx")

    def __init__(self, size: int, device: torch.device, dtype: torch.dtype = torch.float64):
        self.buf = torch.zeros(size, dtype=dtype, device=device)
        self.idx = 0

    def add(self, value: torch.Tensor):
        self.buf[self.idx] = value
        self.idx = (self.idx + 1) % self.buf.numel()

    def view(self) -> torch.Tensor:
        i = self.idx
        return torch.cat((self.buf[i:], self.buf[:i]), dim=0)


class OnlineOLS:
    """Online 2-var OLS: y = β0 + β1 * x, with sliding window of size m."""
    __slots__ = ("ring_x", "ring_y", "Sxx", "Sx", "Sxy", "Sy", "n")

    def __init__(self, m: int, device: torch.device):
        self.ring_x = Ring(m, device)
        self.ring_y = Ring(m, device)
        self.Sxx = torch.tensor(0.0, device=device)
        self.Sx = torch.tensor(0.0, device=device)
        self.Sxy = torch.tensor(0.0, device=device)
        self.Sy = torch.tensor(0.0, device=device)
        self.n = 0

    def add(self, x: torch.Tensor, y: torch.Tensor):
        if self.n == self.ring_x.buf.numel():
            x_old = self.ring_x.buf[self.ring_x.idx]
            y_old = self.ring_y.buf[self.ring_y.idx]
            self.Sxx -= x_old * x_old
            self.Sx -= x_old
            self.Sxy -= x_old * y_old
            self.Sy -= y_old
        else:
            self.n += 1

        self.ring_x.add(x)
        self.ring_y.add(y)
        self.Sxx += x * x
        self.Sx += x
        self.Sxy += x * y
        self.Sy += y

    def coef(self) -> Tuple[torch.Tensor, torch.Tensor]:
        det = self.n * self.Sxx - self.Sx * self.Sx
        if det.abs() < 1e-12:
            return torch.tensor(0.0, device=self.Sxx.device), torch.tensor(0.0, device=self.Sxx.device)
        beta1 = (self.n * self.Sxy - self.Sx * self.Sy) / det
        beta0 = (self.Sy - beta1 * self.Sx) / self.n
        return beta1, beta0


# -----------------------------------------------------------------------------
# 4. Adaptive Market Maker
# -----------------------------------------------------------------------------
class AdaptiveMarketMaker:
    """Vectorised market-maker with O(1) updates using online OLS."""
    def __init__(self, cfg: Config):
        device = torch.device(cfg.device)
        self.theta = torch.tensor(cfg.theta, device=device)
        # two OLS trackers
        self.ols_zp = OnlineOLS(cfg.Tm, device)
        self.ols_yv = OnlineOLS(cfg.Tm, device)

    def determine_price(self, yt: torch.Tensor) -> torch.Tensor:
        xi1, _ = self.ols_zp.coef()
        gamma1, g0 = self.ols_yv.coef()
        lam = (xi1 + self.theta * gamma1) / (xi1**2 + self.theta)
        return g0 + lam * yt

    def update(self, vt: torch.Tensor, pt: torch.Tensor, zt: torch.Tensor, yt: torch.Tensor):
        vt = vt.flatten()
        pt = pt.flatten()
        zt = zt.flatten()
        yt = yt.flatten()
        for x, y in zip(zt, pt):
            self.ols_zp.add(x, y)
        for x, y in zip(yt, vt):
            self.ols_yv.add(x, y)


# -----------------------------------------------------------------------------
# 5. Initialise Q batch
# -----------------------------------------------------------------------------
def initialise_Q_batch(cfg: Config,
                        p_disc: torch.Tensor,
                        v_disc: torch.Tensor,
                        x_disc: torch.Tensor,
                        B: int) -> torch.Tensor:
    """Create initial Q-table tensor of shape (B, I, Np, Nv, Nx)."""
    device = torch.device(cfg.device)
    I, Np, Nv, Nx = cfg.I, cfg.Np, cfg.Nv, cfg.Nx
    Sx = x_disc.sum()
    base_v = v_disc - cfg.v_bar - cfg.lambda_N * cfg.chi_N
    const_v = (Nx * base_v - cfg.lambda_N * (I - 1) * Sx) / ((1 - cfg.rho) * Nx)
    core = const_v.view(1, 1, 1, Nv, 1) * x_disc.view(1, 1, 1, 1, Nx)
    Q0 = core.expand(B, I, Np, Nv, Nx).to(torch.float32)
    return Q0


# -----------------------------------------------------------------------------
# 6. Simulation driver
# -----------------------------------------------------------------------------
def simulate(cfg: Config, out_path: Path):
    device = torch.device(cfg.device)
    # build grids\    
    p_disc, v_disc, x_disc = build_discretisations(cfg)

    # allocate tensors
    B, T = cfg.batch, cfg.steps
    visit_count = torch.zeros((B, cfg.I, cfg.Np, cfg.Nv), dtype=torch.int32, device=device)
    Q_table = initialise_Q_batch(cfg, p_disc, v_disc, x_disc, B)
    last_opt = -torch.ones((B, cfg.I, cfg.Np, cfg.Nv), dtype=torch.int16, device=device)
    conv_ctr = torch.zeros((B, cfg.I), dtype=torch.int32, device=device)

    # helper indices
    batch_ix = torch.arange(B, device=device).unsqueeze(1)  # (B,1)
    agent_ix = torch.arange(cfg.I, device=device).unsqueeze(0)  # (1,I)

    # random paths and noise
    noise = torch.randn((B, T), device=device) * cfg.sigma_u
    v_path = torch.randint(0, cfg.Nv, (B, T), device=device)

    # initial indices
    p_idx = torch.randint(0, cfg.Np, (B,), device=device)
    v_idx = v_path[:, 0]

    # market maker
    mm = AdaptiveMarketMaker(cfg)

    # initial greedy memory
    greedy = Q_table[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1)].argmax(dim=-1)
    memory = torch.take_along_dim(
        Q_table[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1)],
        greedy.unsqueeze(-1), dim=-1
    ).squeeze(-1)

    # main loop
    for t in range(T - 1):
        u_t = noise[:, t]
        # ε-greedy exploration
        eps = torch.exp(-cfg.beta * visit_count[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1)])
        visit_count[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1)] += 1
        explore = torch.rand((B, cfg.I), device=device) < eps
        greedy = Q_table[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1)].argmax(dim=-1)
        randomA = torch.randint(0, cfg.Nx, (B, cfg.I), device=device)
        action = torch.where(explore, randomA, greedy)

        # trades & mm update
        x_sel = x_disc[action]  # (B,I)
        y_sum = x_sel.sum(dim=1) + u_t
        p_val = mm.determine_price(y_sum)
        z_val = cfg.xi * (p_val - cfg.v_bar)
        mm.update(v_disc[v_idx], p_val, z_val, y_sum)

        # rewards & Bellman
        profit = x_sel * (v_disc[v_idx].unsqueeze(1) - p_val.unsqueeze(1))
        best = Q_table[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1)].max(dim=-1).values
        bellman = (1 - cfg.alpha) * memory + cfg.alpha * (profit + cfg.rho * best)

        # write back Q
        Q_table[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1), action] = bellman
        memory = bellman

        # convergence
        changed = greedy != last_opt[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1)]
        reset = changed.any(dim=1)
        conv_ctr[reset] = 0
        conv_ctr[~reset] += 1
        last_opt[batch_ix, agent_ix, p_idx.unsqueeze(1), v_idx.unsqueeze(1)] = greedy

        # next state indices
        v_idx = v_path[:, t + 1]
        # find closest price index
        diff = torch.abs(p_disc.unsqueeze(1) - p_val.unsqueeze(0))  # (Np,B)
        p_idx = diff.argmin(dim=0)

    # save Q-table
    torch.save(Q_table.cpu(), str(out_path))


# -----------------------------------------------------------------------------
# 7. CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--out", required=True, help="Path to save Q-table (.pt)")
    args = parser.parse_args()
    cfg = Config(batch=args.batch, steps=args.steps)
    simulate(cfg, Path(args.out))