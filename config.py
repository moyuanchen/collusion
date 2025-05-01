from dataclasses import dataclass
from agents import solve_chiM, solve_chiN
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


@dataclass
class Config:
    # Environment Parameters
    I: int = 2              # Number of Collusive Agents
    v_bar: float = 1        # Mean Value of the security
    sigma_v: float = 1      # Standard deviation of the security value

    # Discretization Parameters
    Np: int = 31            # Discretization of the price
    Nv: int = 10            # Discretization of the value
    Nx: int = 15            # Discretization of the action
    iota: float = 0.1       # Elastic Range

    # Informed Agents Parameters
    rho: float = 0.95       # Discount factor
    alpha: float = 0.01     # Learning rate (forgetting rate)
    beta: float = 1e-5      # exploration rate decay

    # Preferred Habitate Agents Parameters
    xi: float = 500         # Risk aversion parameter

    # Adapetive Market Maker Agent Parameters
    theta: float= 0.1       # Market maker's pricing error risk aversion
    Tm: int = 10000         # Market maker's time horizon

    # Noise Traders Parameters
    u_bar: float = 0        # Mean noise trade volume
    sigma_u: float = 0.1      # Standard deviation of the noise trade volume

    # Calculate lambda and chi using solve_chiM and solve_chiN
    # chi_M: float
    # lambda_M: float = sigma_u / sigma_v
    # chi_N: float
    # lambda_N: float = np.sqrt(I) * sigma_u / ((I+1) * sigma_v)
    chi_M, lambda_M = solve_chiM(I, xi, sigma_u, sigma_v, theta, tol=1e-12, max_iter=10000)
    chi_N, lambda_N = solve_chiN(I, xi, sigma_u, sigma_v, theta, tol=1e-12, max_iter=10000)

