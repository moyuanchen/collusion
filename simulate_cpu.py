#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import torch
import torch.multiprocessing as mp
# -----------------------------------------------------------------------------
# 0. Calculate chi_M and chi_N
# -----------------------------------------------------------------------------

def solve_chiN(I, xi, sigma_u, sigma_v, theta, tol=1e-12, max_iter=10000):
    """
    Solve for the noncollusive slope chi^N in the Kyle-type model.

    We use the 3-equation system from Section 3.2 in the paper:
      (1) chi^N = 1 / [(I+1)*lambda^N]
      (2) lambda^N = [theta * gamma^N + xi] / (theta + xi^2)
      (3) gamma^N = (I*chi^N) / [(I*chi^N)^2 + (sigma_u/sigma_v)^2]

    We do simple fixed-point iteration over chi^N.

    Returns:
      float: chi^N
      float: lambda^N
    """
    chi = 0.1  # Arbitrary initial guess
    for _ in range(max_iter):
        # Given chi, compute gamma^N:
        gamma = (I * chi) / ((I * chi)**2 + (sigma_u / sigma_v)**2)

        # Then lambda^N:
        lam = (theta * gamma + xi) / (theta + xi**2)

        # Then the updated chi^N:
        new_chi = 1.0 / ((I + 1) * lam)
        # print(new_chi)
        if abs(new_chi - chi) < tol:
            return new_chi, lam
        chi = new_chi

    raise RuntimeError("solve_chiN did not converge within max_iter")

def solve_chiM(I, xi, sigma_u, sigma_v, theta, tol=1e-12, max_iter=10000):
    """
    Solve for the *perfect-collusion* slope chi^M in the Kyle-type model.

    From Section 3.3 in the paper:
      (1) chi^M = 1 / [2*I * lambda^M]
      (2) lambda^M = [theta * gamma^M + xi] / (theta + xi^2)
      (3) gamma^M = (I*chi^M) / [(I*chi^M)^2 + (sigma_u/sigma_v)^2]

    Similar fixed-point iteration as above.
    """
    chi = 0.1  # Arbitrary initial guess
    for _ in range(max_iter):
        gamma = (I * chi) / ((I * chi)**2 + (sigma_u / sigma_v)**2)
        lam = (theta * gamma + xi) / (theta + xi**2)
        new_chi = 1.0 / (2.0 * I * lam)
        # print(new_chi)
        if abs(new_chi - chi) < tol:
            return new_chi, lam
        chi = new_chi

    raise RuntimeError("solve_chiM did not converge within max_iter")


# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # model dimensions
    I: int = 2
    batch: int = 1_000 # Total batch size
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

    # OLS window and MM aggressiveness
    Tm: int = 10_000 # Ring buffer size for OLS
    theta: float = 0.1

    # device for torch tensors
    device: str = "cpu"
    num_workers: int = 1 # Number of CPU worker processes
    # state-space parameters
    chi_N_, lambda_N_ = solve_chiN(I, xi, sigma_u, sigma_v, theta)
    chi_M_, lambda_M_ = solve_chiM(I, xi, sigma_u, sigma_v, theta)
    chi_N: float = chi_N_
    chi_M: float = chi_M_
    lambda_N: float = lambda_N_
    lambda_M: float = lambda_M_



# -----------------------------------------------------------------------------
# 2. Discretisation helpers
# -----------------------------------------------------------------------------
def _grid_linspace(low: float, high: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(low, high, n, dtype=dtype, device=device)

def build_discretisations(cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(cfg.device)
    probs = (2 * torch.arange(1, cfg.Nv + 1, device=device, dtype=torch.float32) - 1) / (2 * cfg.Nv)
    v_disc = cfg.v_bar + cfg.sigma_v * torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(2 * probs - 1)
    span_x = abs(cfg.chi_N - cfg.chi_M)
    low_x = -max(cfg.chi_N, cfg.chi_M) - cfg.iota * span_x
    high_x = max(cfg.chi_N, cfg.chi_M) + cfg.iota * span_x
    x_disc = _grid_linspace(low_x, high_x, cfg.Nx, device, torch.float32)
    lam_max = max(cfg.lambda_N, cfg.lambda_M)
    ph = cfg.v_bar + lam_max * (cfg.I * max(cfg.chi_N, cfg.chi_M) + cfg.sigma_u * 1.96)
    pl = cfg.v_bar - lam_max * (cfg.I * max(cfg.chi_N, cfg.chi_M) + cfg.sigma_u * 1.96)
    span_p = ph - pl
    p_disc = _grid_linspace(pl - cfg.iota * span_p, ph + cfg.iota * span_p, cfg.Np, device, torch.float32)
    return p_disc, v_disc, x_disc

# -----------------------------------------------------------------------------
# 3. Vectorized Ring buffer & Online OLS
# -----------------------------------------------------------------------------
class VectorizedRing:
    """Circular buffer storing a fixed number of scalars for a batch of series."""
    __slots__ = ("buf", "idx", "batch_size", "ring_size", "device", "dtype")

    def __init__(self, batch_size: int, ring_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.batch_size = batch_size
        self.ring_size = ring_size
        self.device = device
        self.dtype = dtype
        self.buf = torch.zeros((batch_size, ring_size), dtype=dtype, device=device)
        self.idx = torch.zeros(batch_size, dtype=torch.long, device=device) # Current index to write to for each batch element

    def add(self, values: torch.Tensor):
        """Add a batch of values, one for each series."""
        if values.shape[0] != self.batch_size:
            raise ValueError(f"Input values batch size ({values.shape[0]}) must match Ring batch size ({self.batch_size})")
        
        # Scatter add: self.buf[batch_indices, self.idx] = values
        # Equivalent:
        batch_indices = torch.arange(self.batch_size, device=self.device)
        self.buf[batch_indices, self.idx] = values
        self.idx = (self.idx + 1) % self.ring_size

    def get_oldest_values(self) -> torch.Tensor:
        """Get the oldest values that are about to be overwritten, for each series."""
        # The oldest value is at self.idx because that's where the next write will occur
        batch_indices = torch.arange(self.batch_size, device=self.device)
        return self.buf[batch_indices, self.idx]

class VectorizedOnlineOLS:
    """Vectorized Online 2-var OLS: y_i = β0_i + β1_i * x_i for a batch of i series."""
    __slots__ = ("ring_x", "ring_y", "Sxx", "Sx", "Sxy", "Sy", "n", "batch_size", "ring_size", "device")

    def __init__(self, batch_size: int, ring_size: int, device: torch.device):
        self.batch_size = batch_size
        self.ring_size = ring_size # This is 'm' from the original OnlineOLS
        self.device = device

        self.ring_x = VectorizedRing(batch_size, ring_size, device, dtype=torch.float32)
        self.ring_y = VectorizedRing(batch_size, ring_size, device, dtype=torch.float32)
        
        self.Sxx = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.Sx = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.Sxy = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.Sy = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.n = torch.zeros(batch_size, dtype=torch.float32, device=device) # Using float for potential fractional counts if averaging, but long/int if strict counting

    def add(self, x: torch.Tensor, y: torch.Tensor):
        """Add a batch of (x,y) pairs, one for each OLS series."""
        if x.shape[0] != self.batch_size or y.shape[0] != self.batch_size:
            raise ValueError("Input x and y batch sizes must match OLS batch size.")

        # Identify which series are at full capacity
        full_capacity_mask = (self.n == self.ring_size)
        
        if torch.any(full_capacity_mask):
            x_old = self.ring_x.get_oldest_values()
            y_old = self.ring_y.get_oldest_values()
            
            self.Sxx[full_capacity_mask] -= x_old[full_capacity_mask] * x_old[full_capacity_mask]
            self.Sx[full_capacity_mask] -= x_old[full_capacity_mask]
            self.Sxy[full_capacity_mask] -= x_old[full_capacity_mask] * y_old[full_capacity_mask]
            self.Sy[full_capacity_mask] -= y_old[full_capacity_mask]
        
        # For series not yet at full capacity, increment count
        not_full_capacity_mask = ~full_capacity_mask
        self.n[not_full_capacity_mask] += 1

        self.ring_x.add(x)
        self.ring_y.add(y)
        
        self.Sxx += x * x
        self.Sx += x
        self.Sxy += x * y
        self.Sy += y

    def coef(self) -> Tuple[torch.Tensor, torch.Tensor]:
        beta1 = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        beta0 = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)

        # Calculate for series with enough data points (n >= 2)
        valid_mask = (self.n >= 2)
        if not torch.any(valid_mask):
            return beta1, beta0

        n_v = self.n[valid_mask]
        Sxx_v = self.Sxx[valid_mask]
        Sx_v = self.Sx[valid_mask]
        Sxy_v = self.Sxy[valid_mask]
        Sy_v = self.Sy[valid_mask]

        det = n_v * Sxx_v - Sx_v * Sx_v
        
        # Avoid division by zero for singular cases
        # Coefficients will remain 0 for these cases as initialized
        calc_mask = det.abs() >= 1e-12 
        
        if torch.any(calc_mask):
            det_c = det[calc_mask]
            n_c = n_v[calc_mask]
            Sxy_c = Sxy_v[calc_mask]
            Sx_c = Sx_v[calc_mask]
            Sy_c = Sy_v[calc_mask]

            beta1_v = torch.zeros_like(det)
            beta0_v = torch.zeros_like(det)

            beta1_v[calc_mask] = (n_c * Sxy_c - Sx_c * Sy_c) / det_c
            beta0_v[calc_mask] = (Sy_c - beta1_v[calc_mask] * Sx_c) / n_c
            
            # Place calculated coefficients back into the main tensors
            # Need to map indices from valid_mask and calc_mask correctly
            valid_indices = torch.where(valid_mask)[0]
            calc_indices_within_valid = torch.where(calc_mask)[0]
            final_calc_indices = valid_indices[calc_indices_within_valid]

            beta1[final_calc_indices] = beta1_v[calc_mask]
            beta0[final_calc_indices] = beta0_v[calc_mask]
            
        return beta1, beta0

# -----------------------------------------------------------------------------
# 4. Vectorized Adaptive Market Maker
# -----------------------------------------------------------------------------
class VectorizedAdaptiveMarketMaker:
    """Vectorized market-maker, managing a batch of independent OLS trackers."""
    def __init__(self, cfg: Config, batch_size: int, device: torch.device):
        self.theta = torch.tensor(cfg.theta, device=device, dtype=torch.float32)
        self.ols_zp = VectorizedOnlineOLS(batch_size, cfg.Tm, device)
        self.ols_yv = VectorizedOnlineOLS(batch_size, cfg.Tm, device)
        self.device = device
        self.batch_size = batch_size # Store batch_size

    def determine_price(self, yt: torch.Tensor) -> torch.Tensor:
        # yt is shape (batch_size,)
        if yt.shape[0] != self.batch_size:
            raise ValueError(f"Input yt batch size ({yt.shape[0]}) must match MM batch size ({self.batch_size})")

        xi1, _ = self.ols_zp.coef()  # xi1 is shape (batch_size,)
        gamma1, g0 = self.ols_yv.coef() # gamma1, g0 are shape (batch_size,)

        lam = torch.zeros_like(xi1) # shape (batch_size,)
        denominator = (xi1**2 + self.theta)
        
        # Calculate lam only where denominator is not too small
        calc_mask = torch.abs(denominator) >= 1e-12
        lam[calc_mask] = (xi1[calc_mask] + self.theta * gamma1[calc_mask]) / denominator[calc_mask]
        
        return g0 + lam * yt

    def update(self, vt: torch.Tensor, pt: torch.Tensor, zt: torch.Tensor, yt: torch.Tensor):
        # All inputs are shape (batch_size,)
        if not all(tensor.shape[0] == self.batch_size for tensor in [vt, pt, zt, yt]):
            raise ValueError(f"All input tensor batch sizes must match MM batch size ({self.batch_size})")

        self.ols_zp.add(zt, pt)
        self.ols_yv.add(yt, vt)
    def state_dict(self):
        return {
            'theta': self.theta.cpu(),
            'zp': {
                'Sxx': self.ols_zp.Sxx.cpu(),
                'Sx': self.ols_zp.Sx.cpu(),
                'Sxy': self.ols_zp.Sxy.cpu(),
                'Sy': self.ols_zp.Sy.cpu(),
                'n': self.ols_zp.n.cpu(),
                'ring_x': self.ols_zp.ring_x.buf.cpu(),
                'ring_x_idx': self.ols_zp.ring_x.idx.cpu(),
                'ring_y': self.ols_zp.ring_y.buf.cpu(),
                'ring_y_idx': self.ols_zp.ring_y.idx.cpu(),
            },
            'yv': {
                'Sxx': self.ols_yv.Sxx.cpu(),
                'Sx': self.ols_yv.Sx.cpu(),
                'Sxy': self.ols_yv.Sxy.cpu(),
                'Sy': self.ols_yv.Sy.cpu(),
                'n': self.ols_yv.n.cpu(),
                'ring_x': self.ols_yv.ring_x.buf.cpu(),
                'ring_x_idx': self.ols_yv.ring_x.idx.cpu(),
                'ring_y': self.ols_yv.ring_y.buf.cpu(),
                'ring_y_idx': self.ols_yv.ring_y.idx.cpu(),
            }
        }

    @classmethod
    def load_state_dict(cls, cfg: Config, state: dict, device: torch.device):
        batch_size = state['zp']['Sxx'].shape[0]
        maker = cls(cfg, batch_size, device)
        maker.theta = state['theta'].to(device)
        # load zp
        zp = state['zp']
        maker.ols_zp.Sxx = zp['Sxx'].to(device)
        maker.ols_zp.Sx = zp['Sx'].to(device)
        maker.ols_zp.Sxy = zp['Sxy'].to(device)
        maker.ols_zp.Sy = zp['Sy'].to(device)
        maker.ols_zp.n = zp['n'].to(device)
        maker.ols_zp.ring_x.buf = zp['ring_x'].to(device)
        maker.ols_zp.ring_x.idx = zp['ring_x_idx'].to(device)
        maker.ols_zp.ring_y.buf = zp['ring_y'].to(device)
        maker.ols_zp.ring_y.idx = zp['ring_y_idx'].to(device)
        # load yv
        yv = state['yv']
        maker.ols_yv.Sxx = yv['Sxx'].to(device)
        maker.ols_yv.Sx = yv['Sx'].to(device)
        maker.ols_yv.Sxy = yv['Sxy'].to(device)
        maker.ols_yv.Sy = yv['Sy'].to(device)
        maker.ols_yv.n = yv['n'].to(device)
        maker.ols_yv.ring_x.buf = yv['ring_x'].to(device)
        maker.ols_yv.ring_x.idx = yv['ring_x_idx'].to(device)
        maker.ols_yv.ring_y.buf = yv['ring_y'].to(device)
        maker.ols_yv.ring_y.idx = yv['ring_y_idx'].to(device)
        return maker

# -----------------------------------------------------------------------------
# 5. Initialise Q batch
# -----------------------------------------------------------------------------
def initialise_Q_batch(cfg: Config,
                        p_disc: torch.Tensor,
                        v_disc: torch.Tensor,
                        x_disc: torch.Tensor,
                        B: int, 
                        init_device: torch.device) -> torch.Tensor:
    I, Np, Nv, Nx = cfg.I, cfg.Np, cfg.Nv, cfg.Nx
    v_disc_init = v_disc.to(init_device)
    x_disc_init = x_disc.to(init_device)
    Sx = x_disc_init.sum()
    base_v = v_disc_init - cfg.v_bar - cfg.lambda_N * cfg.chi_N
    const_v = (Nx * base_v - cfg.lambda_N * (I - 1) * Sx) / ((1 - cfg.rho) * Nx)
    core = const_v.view(1, 1, 1, Nv, 1) * x_disc_init.view(1, 1, 1, 1, Nx)
    Q0 = core.expand(B, I, Np, Nv, Nx).to(dtype=torch.float32, device=init_device)
    return Q0

# -----------------------------------------------------------------------------
# 6. Simulation worker
# -----------------------------------------------------------------------------
def worker_fn(rank: int, cfg: Config,
              q_table_shared: torch.Tensor,
              visit_count_shared: torch.Tensor,
              last_opt_shared: torch.Tensor,
              conv_ctr_shared: torch.Tensor,
              profit_hist_shared: torch.Tensor,
              p_disc: torch.Tensor, v_disc: torch.Tensor, x_disc: torch.Tensor,
              noise_all: torch.Tensor, v_path_all: torch.Tensor,
              mm: VectorizedAdaptiveMarketMaker,
              out_path: Path):

    worker_device = torch.device(cfg.device) 
    torch.manual_seed(torch.initial_seed() + rank) 

    total_batch_size = cfg.batch
    num_workers = cfg.num_workers

    items_per_worker = total_batch_size // num_workers
    start_idx = rank * items_per_worker
    end_idx = (rank + 1) * items_per_worker
    if rank == num_workers - 1: 
        end_idx = total_batch_size
    
    current_batch_size = end_idx - start_idx
    if current_batch_size == 0:
        return 

    q_table = q_table_shared[start_idx:end_idx]
    visit_count = visit_count_shared[start_idx:end_idx]
    last_opt = last_opt_shared[start_idx:end_idx]
    conv_ctr = conv_ctr_shared[start_idx:end_idx]
    profit_hist = profit_hist_shared[start_idx:end_idx]
    noise = noise_all[start_idx:end_idx, :].to(worker_device) # Move worker's slice to its device
    v_path = v_path_all[start_idx:end_idx, :].to(worker_device)

    p_disc = p_disc.to(worker_device)
    v_disc = v_disc.to(worker_device)
    x_disc = x_disc.to(worker_device)

    batch_ix = torch.arange(current_batch_size, device=worker_device).unsqueeze(1)
    agent_ix = torch.arange(cfg.I, device=worker_device).unsqueeze(0)  

    p_idx = torch.randint(0, cfg.Np, (current_batch_size,), device=worker_device)
    v_idx = v_path[:, 0] 

    # Each worker gets a VECTORIZED MM that handles `current_batch_size` independent simulations
    # mm = VectorizedAdaptiveMarketMaker(cfg, current_batch_size, worker_device)

    p_idx_exp = p_idx.unsqueeze(1) 
    v_idx_exp = v_idx.unsqueeze(1) 
    
    q_slice_for_init = q_table[batch_ix, agent_ix, p_idx_exp, v_idx_exp] 
    greedy = q_slice_for_init.argmax(dim=-1) 
    
    memory = torch.take_along_dim(
        q_slice_for_init,
        greedy.unsqueeze(-1), dim=-1
    ).squeeze(-1) 

    for t in range(cfg.steps - 1):
        if rank == 0 and t % 10000 == 0: 
            print(f"Rank 0, Step: {t}/{cfg.steps}")

        u_t = noise[:, t] 

        current_visit_counts = visit_count[batch_ix, agent_ix, p_idx_exp, v_idx_exp]
        eps = torch.exp(-cfg.beta * current_visit_counts)
        visit_count[batch_ix, agent_ix, p_idx_exp, v_idx_exp] += 1
        
        explore = torch.rand((current_batch_size, cfg.I), device=worker_device) < eps
        
        q_slice_for_action = q_table[batch_ix, agent_ix, p_idx_exp, v_idx_exp] 
        greedy = q_slice_for_action.argmax(dim=-1) 
        
        randomA = torch.randint(0, cfg.Nx, (current_batch_size, cfg.I), device=worker_device)
        action = torch.where(explore, randomA, greedy) 

        x_sel = x_disc[action]  
        y_sum = x_sel.sum(dim=1) + u_t 
        
        p_val = mm.determine_price(y_sum.to(torch.float32)) 
        z_val = cfg.xi * (p_val - cfg.v_bar) 

        mm.update(v_disc[v_idx].to(torch.float32), 
                  p_val.to(torch.float32), 
                  z_val.to(torch.float32), 
                  y_sum.to(torch.float32))

        profit = x_sel * (v_disc[v_idx].unsqueeze(1) - p_val.unsqueeze(1))  # shape (current_batch_size, cfg.I)
        profit_hist[:, :, t] = profit 
        q_slice_for_best = q_table[batch_ix, agent_ix, p_idx_exp, v_idx_exp] 
        best_q_values = q_slice_for_best.max(dim=-1).values 
        
        bellman = (1 - cfg.alpha) * memory + cfg.alpha * (profit + cfg.rho * best_q_values) 

        idx_b = batch_ix.expand(-1, cfg.I) 
        idx_i = agent_ix.expand(current_batch_size, -1) 
        idx_p = p_idx_exp.expand(-1, cfg.I) 
        idx_v = v_idx_exp.expand(-1, cfg.I) 
        
        q_table[idx_b.clone(), idx_i.clone(), idx_p.clone(), idx_v.clone(), action.clone()] = bellman
        memory = bellman 

        current_last_opt = last_opt[batch_ix, agent_ix, p_idx_exp, v_idx_exp]
        changed = greedy != current_last_opt 
        
        reset_local = changed.any(dim=1) 
        
        conv_ctr[reset_local, :] = 0
        conv_ctr[~reset_local, :] += 1 
        
        last_opt[batch_ix, agent_ix, p_idx_exp, v_idx_exp] = greedy.to(last_opt.dtype)

        v_idx = v_path[:, t + 1] 
        v_idx_exp = v_idx.unsqueeze(1) 

        diff = torch.abs(p_disc.unsqueeze(1) - p_val.to(p_disc.dtype).unsqueeze(0)) 
        p_idx = diff.argmin(dim=0) 
        p_idx_exp = p_idx.unsqueeze(1) 

    if rank == 0:
        print(f"Worker 0 finished simulation loop.")

# -----------------------------------------------------------------------------
# 7. Simulation driver
# -----------------------------------------------------------------------------
def simulate(cfg: Config, out_path: Path, load_path: Path = None):
    main_process_device = torch.device(cfg.device) 
    print(f"Main process using device: {main_process_device}")
    print(f"Number of workers: {cfg.num_workers}")
    print(f"Total batch size: {cfg.batch}, Steps: {cfg.steps}")
    init_device_for_shared = torch.device("cpu") if cfg.num_workers > 1 else main_process_device
    if load_path:
        print(f"Loading state from {load_path}")
        saved = torch.load(load_path, map_location=main_process_device)
        cfg = Config(**saved['config'])
        q_table_shared = saved['q_table'].to(main_process_device)
        visit_count_shared = saved['visit_count'].to(main_process_device)
        last_opt_shared = saved['last_opt'].to(main_process_device)
        conv_ctr_shared = saved['conv_ctr'].to(main_process_device)
        profit_hist_shared = saved['profit_hist'].to(main_process_device)
        mm = VectorizedAdaptiveMarketMaker.load_state_dict(cfg, saved['marketmaker'], main_process_device)
        print("State loaded successfully.")
        

    if cfg.num_workers > 1 and cfg.device == "cuda":
        print("Warning: Using CUDA with num_workers > 1. Make sure CUDA is properly set up for multiprocessing.")
        # For CUDA with multiprocessing, 'spawn' start method is crucial.
        # Each process will need to initialize its CUDA context on its assigned GPU if multi-GPU.
        # If single GPU, all workers share it, which can lead to contention unless managed.
        # For simplicity with vectorized MM, often num_workers=1 is fine if batch is large and processed on GPU.

    p_disc_main, v_disc_main, x_disc_main = build_discretisations(cfg)

    B = cfg.batch 
    # For shared tensors, they must be on CPU if using mp.spawn with CPU workers primarily.
    # If workers are CUDA-based, sharing strategies are more complex.
    # Assuming CPU workers for now, or num_workers=1
    if load_path is None:
        
        
        q_table_shared = initialise_Q_batch(cfg, p_disc_main, v_disc_main, x_disc_main, B, init_device_for_shared)
        visit_count_shared = torch.zeros((B, cfg.I, cfg.Np, cfg.Nv), dtype=torch.int32, device=init_device_for_shared)
        last_opt_shared = -torch.ones((B, cfg.I, cfg.Np, cfg.Nv), dtype=torch.int32, device=init_device_for_shared)
        conv_ctr_shared = torch.zeros((B, cfg.I), dtype=torch.int32, device=init_device_for_shared) 
        profit_hist_shared = torch.zeros((B, cfg.I, cfg.steps),
                                        dtype=torch.float32,
                                        device=init_device_for_shared)
        mm = VectorizedAdaptiveMarketMaker(cfg, B, main_process_device)
    noise_all_main = torch.randn((B, cfg.steps), device=init_device_for_shared, dtype=torch.float32) * torch.tensor(cfg.sigma_u, dtype=torch.float32)
    v_path_all_main = torch.randint(0, cfg.Nv, (B, cfg.steps), device=init_device_for_shared)

    if cfg.num_workers > 1:
        profit_hist_shared.share_memory_()
        q_table_shared.share_memory_()
        visit_count_shared.share_memory_()
        last_opt_shared.share_memory_()
        conv_ctr_shared.share_memory_()
        # noise_all_main and v_path_all_main will be copied to each process by spawn,
        # but ensure they are on CPU if workers are CPU based.
        # If target cfg.device is CUDA and num_workers=1, these can stay on CUDA.
        # The worker_fn will move its slice of noise/v_path to cfg.device.

    if cfg.num_workers > 1:
        # args_tuple = (cfg, q_table_shared, visit_count_shared, last_opt_shared, conv_ctr_shared, profit_hist_shared,
        #         p_disc_main.clone(), v_disc_main.clone(), x_disc_main.clone(), # Pass clones if they might be modified or for safety
        #         noise_all_main, v_path_all_main, out_path)
        # print("Starting worker processes...")
        # mp.spawn(worker_fn,
        #          args=args_tuple,
        #          nprocs=cfg.num_workers,
        #          join=True)
        # print("All workers finished.")
        # torch.save({
        #     'q_table': q_table_shared.cpu(),
        #     'conv_ctr': conv_ctr_shared.cpu(),
        #     'profit_hist': profit_hist_shared.cpu(),
        # }, str(out_path))
        print("Multiprocessing not implemented in this version.")
        pass
    else: # Run in a single process (no spawning)
        print("Running in a single process...")
        # The "rank" is 0, and all data is local.
        # We directly call worker_fn, but it expects shared tensors.
        # For num_workers=1, we can just pass the tensors directly.
        # The worker_fn will use cfg.device.
        
        # Ensure data for single worker is on the configured cfg.device
        q_table_local = q_table_shared.to(main_process_device)
        visit_count_local = visit_count_shared.to(main_process_device)
        last_opt_local = last_opt_shared.to(main_process_device)
        conv_ctr_local = conv_ctr_shared.to(main_process_device)
        profit_hist_local = profit_hist_shared.to(main_process_device)
        
        p_disc_local = p_disc_main.to(main_process_device)
        v_disc_local = v_disc_main.to(main_process_device)
        x_disc_local = x_disc_main.to(main_process_device)
        noise_all_local = noise_all_main.to(main_process_device)
        v_path_all_local = v_path_all_main.to(main_process_device)

        worker_fn(0, cfg, q_table_local, visit_count_local, last_opt_local, conv_ctr_local, profit_hist_local,
                  p_disc_local, v_disc_local, x_disc_local,
                  noise_all_local, v_path_all_local, mm, out_path)
        # Assume profit_history was recorded during simulation (e.g., as a tensor or list)
        # If it's not already on CPU, move it to CPU before saving
        # q_table_shared = q_table_local  # Update reference for saving
        torch.save({
            'q_table': q_table_local.cpu(),
            'visit_count': visit_count_local.cpu(),
            'last_opt': last_opt_local.cpu(),
            'conv_ctr': conv_ctr_local.cpu(),
            'profit_hist': profit_hist_local.cpu(),
            'marketmaker': mm.state_dict(),
            'config': cfg.__dict__
        }, str(out_path))
    print(f"Q-table saved to {out_path}")
    return conv_ctr_local

# -----------------------------------------------------------------------------
# 8. CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=100, help="Total batch size for simulation.") # Reduced for quick test
    parser.add_argument("--steps", type=int, default=1000) # Reduced for quicker testing
    parser.add_argument("--workers", type=int, default=1, help="Number of CPU worker processes.") # Default to 1 for vectorized MM
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--out", required=True, help="Path to save Q-table (.pt)")
    parser.add_argument("--sigma_u", type=float, default=0.1, help="Override noise-trader σᵤ (float)")
    parser.add_argument("--convergence", type=int, default=1000000, help="Convergence threshold for simulation")
    parser.add_argument("--load", type=str, default=None, help="Path to load state from (.pt)")
    
    cli_args = parser.parse_args()
    
    # For multiprocessing, 'spawn' is generally preferred.
    # This needs to be at the top level of the main module.
    if cli_args.workers > 1 and mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'")
        except RuntimeError as e:
            print(f"Warning: Could not set start method to 'spawn': {e}")


    cfg = Config(batch=cli_args.batch, 
                 steps=cli_args.steps, 
                 num_workers=cli_args.workers,
                 device=cli_args.device,
                 sigma_u=cli_args.sigma_u)
    
    if cfg.num_workers > 1 and cfg.device == "cuda":
        print("Note: Using multiple workers with CUDA. Ensure your setup supports this well.")
        print("Consider if num_workers=1 on a single powerful GPU might be more efficient for fully vectorized operations.")
    elif cfg.num_workers == 1 and cfg.device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            cfg.device = "cpu"
        else:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        
    base = Path(cli_args.out)
    next_save = base.with_name(base.stem + "_a.pt")
    alt_save = base.with_name(base.stem + "_b.pt")

    load = Path(cli_args.load) if cli_args.load else None

    conv = int(simulate(cfg, next_save, load).min().item())
    print(f"Convergence counter: {conv}")
    while conv < cli_args.convergence:
        load = next_save
        next_save, alt_save = alt_save, next_save
        conv = int(simulate(cfg, next_save, load).min().item())
        print(f"Convergence counter: {conv}")

    # print("Simulation complete.")