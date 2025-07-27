from dataclasses import dataclass
# from agents import solve_chiM, solve_chiN
import pickle
import numpy as np
def save(path, config):
    # np.save(path, config)
    with open(path, 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    # Load the configuration dictionary and unpack into a Config object
    with open(path, 'rb') as f:
        config = pickle.load(f)

    return config

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
@dataclass
class Config:
    # model dimensions
    I: int = 2
    batch: int = 100 # Total batch size
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