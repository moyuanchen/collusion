import argparse
from dataclasses import dataclass
from pathlib import Path
import torch

# -----------------------------------------------------------------------------
# 1.  Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # global
    I: int = 2
    batch: int = 1_000
    steps: int = 1_000_000

    # grids
    Np: int = 31
    Nv: int = 10
    Nx: int = 15
    iota: float = 0.1

    # parameters
    v_bar: float = 1.0
    sigma_v: float = 1.0
    sigma_u: float = 0.1

    rho: float = 0.95
    alpha: float = 0.01
    beta: float = 1e-5
    xi: float = 500.0

    chi_N: float = 0.05   # solved earlier – placeholder
    chi_M: float = 0.10
    lambda_N: float = 0.12
    lambda_M: float = 0.18

    Tm: int = 10_000      # MM window
    theta: float = 0.1

    device: str = "cuda"


# -----------------------------------------------------------------------------
# 2.  Discretisation helpers
# -----------------------------------------------------------------------------

def build_grids(cfg: Config):
    dev = cfg.device
    # v grid via inverse CDF
    probs = (2*torch.arange(1, cfg.Nv+1, device=dev) - 1) / (2*cfg.Nv)
    v_disc = cfg.v_bar + cfg.sigma_v * torch.special.ndtri(probs)

    # x grid
    span_x = abs(cfg.chi_N - cfg.chi_M)
    low_x = -max(cfg.chi_N, cfg.chi_M) - cfg.iota*span_x
    high_x = max(cfg.chi_N, cfg.chi_M) + cfg.iota*span_x
    x_disc = torch.linspace(low_x, high_x, cfg.Nx, device=dev)

    # p grid
    lam_max = max(cfg.lambda_N, cfg.lambda_M)
    margin = lam_max * (cfg.I*max(cfg.chi_N, cfg.chi_M) + cfg.sigma_u*1.96)
    ph, pl = cfg.v_bar + margin, cfg.v_bar - margin
    span_p = ph - pl
    p_disc = torch.linspace(pl - cfg.iota*span_p, ph + cfg.iota*span_p, cfg.Np, device=dev)
    return p_disc, v_disc, x_disc


# -----------------------------------------------------------------------------
# 3.  Market Maker (vectorised, O(1) updates)
# -----------------------------------------------------------------------------
class Ring:
    __slots__ = ("buf", "idx")
    def __init__(self, size, device):
        self.buf = torch.zeros(size, device=device)
        self.idx = 0
    def add(self, val):
        self.buf[self.idx] = val
        self.idx = (self.idx + 1) % self.buf.numel()
    def oldest(self):
        return self.buf[self.idx]

class OnlineOLS:
    __slots__ = ("x", "y", "Sxx", "Sx", "Sxy", "Sy", "n", "size")
    def __init__(self, size, device):
        self.x = Ring(size, device)
        self.y = Ring(size, device)
        self.Sxx = self.Sx = self.Sxy = self.Sy = torch.tensor(0.0, device=device)
        self.n = 0
        self.size = size
    def add(self, xt, yt):
        if self.n == self.size:
            xo = self.x.oldest(); yo = self.y.oldest()
            self.Sxx -= xo*xo; self.Sx -= xo; self.Sxy -= xo*yo; self.Sy -= yo
        else:
            self.n += 1
        self.x.add(xt); self.y.add(yt)
        self.Sxx += xt*xt; self.Sx += xt; self.Sxy += xt*yt; self.Sy += yt
    def coef(self):
        det = self.n*self.Sxx - self.Sx**2
        if det == 0:
            return 0.0, 0.0
        b1 = (self.n*self.Sxy - self.Sx*self.Sy)/det
        b0 = (self.Sy - b1*self.Sx)/self.n
        return b1, b0

class MarketMaker:
    def __init__(self, cfg: Config):
        self.theta = cfg.theta
        self.zp = OnlineOLS(cfg.Tm, cfg.device)
        self.yv = OnlineOLS(cfg.Tm, cfg.device)
    def price(self, y):
        xi1, _ = self.zp.coef()
        gamma1, g0 = self.yv.coef()
        lam = (xi1 + self.theta*gamma1) / (xi1**2 + self.theta)
        return g0 + lam*y
    def update(self, v, p, z, y):
        self.zp.add(z, p)
        self.yv.add(y, v)

# -----------------------------------------------------------------------------
# 4.  Simulation driver (batched)
# -----------------------------------------------------------------------------

def simulate(cfg: Config, out_path: Path):
    dev = cfg.device
    B = cfg.batch; T = cfg.steps; I = cfg.I

    p_disc, v_disc, x_disc = build_grids(cfg)

    # RNG
    g = torch.Generator(device=dev)
    g.manual_seed(123)

    # initial indices
    p_idx = torch.randint(0, cfg.Np, (B,), generator=g, device=dev)
    v_idx = torch.randint(0, cfg.Nv, (B,), generator=g, device=dev)

    visit = torch.zeros((I, cfg.Np, cfg.Nv), dtype=torch.int32, device=dev)
    Q     = torch.zeros((I, cfg.Np, cfg.Nv, cfg.Nx), device=dev)

    # track last optimal action for every (i, p, v)
    last_opt = torch.full((cfg.I, cfg.Np, cfg.Nv),
                        -1, dtype=torch.int16, device=dev)

    # convergence counter per agent
    conv_counter = torch.zeros(cfg.I, dtype=torch.int32, device=dev)

    mm = MarketMaker(cfg)

    for t in range(T-1):
        u_noise = torch.randn(B, device=dev) * cfg.sigma_u

        eps   = torch.exp(-cfg.beta * visit[:, p_idx, v_idx])          # (I,B)
        explore = torch.rand(I, B, device=dev) < eps
        greedy  = Q[:, p_idx, v_idx].argmax(-1)
        randA   = torch.randint(0, cfg.Nx, (I,B), generator=g, device=dev)
        act     = torch.where(explore, randA, greedy)                 # (I,B)
        visit[:, p_idx, v_idx] += 1

        # ---------- convergence test ----------
        changed = greedy != last_opt[:, p_idx, v_idx]     # (I,B) bool
        # any change across batch resets that agent’s counter
        reset   = changed.any(dim=1)
        conv_counter[reset] = 0
        conv_counter[~reset] += 1

        # store new optimal actions for next step
        last_opt[:, p_idx, v_idx] = greedy

        # trades
        x_val = x_disc[act]                                           # (I,B)
        y_sum = x_val.sum(dim=0) + u_noise                            # (B,)
        p_val = mm.price(y_sum)                                       # (B,)
        z_val = cfg.xi * (p_val - cfg.v_bar)
        mm.update(v_disc[v_idx], p_val, z_val, y_sum)

        # rewards & Q updates
        reward = x_val * (v_disc[v_idx] - p_val)                      # (I,B)
        best   = Q[:, p_idx, v_idx].amax(-1)
        mem    = Q[:, p_idx, v_idx, act]
        bell   = (1-cfg.alpha)*mem + cfg.alpha*(reward + cfg.rho*best)
        Q[:, p_idx, v_idx, act] = bell

        # advance indices
        v_idx = torch.randint(0, cfg.Nv, (B,), generator=g, device=dev)
        p_idx = torch.cdist(p_val.unsqueeze(1), p_disc.unsqueeze(1)).argmin(-1)

    torch.save(Q.cpu(), out_path)

# -----------------------------------------------------------------------------
# 5.  CLI wrapper
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--out",   required=True)
    cfg = Config()
    cfg.batch = ap.parse_args().batch
    cfg.steps = ap.parse_args().steps
    simulate(cfg, Path(ap.parse_args().out))
