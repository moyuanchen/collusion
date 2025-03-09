from dataclasses import dataclass

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

