import numpy as np
import torch
from itertools import product
from collections import defaultdict
# from util import solve_chiN, solve_chiM
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
class Q_table:

    def __init__(self,
                 config):
        self.Np = range(config.Np)
        self.Nv = range(config.Nv)

        self.states = list(product(self.Np, self.Nv))

        self.Q = {s:np.zeros(config.Nx) for s in self.states}
        
    def get_Q_value(self, state, action):
        return self.Q[state][action]

    def get_best_action(self, state):
        return np.argmax(self.Q[state])

    def get_best_value(self, state):
        return np.max(self.Q[state])

    def update(self, state, action, value):
        self.Q[state][action] = value

class InformedAgent:

    def __init__(self, config):
        
        # parameters
        self.n_actions = config.Nx
        self.Np = config.Np
        self.Nv = config.Nv
        self.n_states = self.Np * self.Nv

        self.rho = config.rho
        self.alpha = config.alpha
        self.beta = config.beta

        self.sigma_v = config.sigma_v
        self.v_bar = config.v_bar
        self.sigma_u = config.sigma_u
        self.xi = config.xi
        self.theta = config.theta
        self.iota = config.iota
        self.I = config.I
        # Q-table for RL
        self.Q = Q_table(config)

        # state count dictionary for epsilon decay
        self.state_count = defaultdict(int)

        # convergence dictionary
        self.last_optimal = {}
        self.convergence_counter = 0

        # discretization of states
        self.get_discrete_states()

        # initialize Q-table
        self.initialize_Q()

    
    def get_epsilon(self, state):
        v = self.state_count[state]
        self.state_count[state] += 1
        return np.exp(-self.beta * v)
    
    def get_action(self, state):
        epsilon = self.get_epsilon(state)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            optimal_action = self.Q.get_best_action(state)
            self.check_convergence(state, optimal_action)
            return optimal_action
        
    def update(self, state, action, reward, next_state):
        learning = self.alpha * (reward + self.rho * self.Q.get_best_value(next_state))
        memory = (1 - self.alpha) * self.Q.get_Q_value(state, action)
        value = learning + memory
        self.Q.update(state, action, value)
    
    def get_grid_point_values_v(self):
        """
        Returns a zero indexed dictionary of the grid points for the state space of v
        """
        standard_normal = torch.distributions.Normal(0, 1)
        grid_point = [(2 * k - 1) / (2 * self.Nv) for k in range(1, self.Nv + 1)]
        values = standard_normal.icdf(torch.tensor(grid_point))
        return {idx: float(self.v_bar + self.sigma_v * value) for idx, value in enumerate(values)}
    
    def get_grid_point_values_x(self):
        self.chiN, self.lambdaN = solve_chiN(I = self.I, xi = self.xi, sigma_u = self.sigma_u, sigma_v = self.sigma_v, theta = self.theta)
        self.chiM, self.lambdaM = solve_chiM(I = self.I, xi = self.xi, sigma_u = self.sigma_u, sigma_v = self.sigma_v, theta = self.theta)
        self.x_n, self.x_m = self.chiN, self.chiM # assuming v - v_bar = 1
        span_x = abs(self.x_n - self.x_m)
        low, high = -max(self.x_n, self.x_m) - self.iota * span_x, max(self.x_n, self.x_m) + self.iota * span_x
        values = np.linspace(low, high, self.n_actions)
        return {idx: float(val) for idx, val in enumerate(values)}
    
    def get_grid_point_values_p(self):
        try:
            lambda_for_p = max(self.lambdaN, self.lambdaM)
        except:
            raise ValueError("Call get_grid_point_values_x() first")
        ph = self.v_bar + lambda_for_p * (self.I * max(self.x_n, self.x_m) + self.sigma_u * 1.96)
        pl = self.v_bar - lambda_for_p * (self.I * max(self.x_n, self.x_m) + self.sigma_u * 1.96)
        span_p = ph - pl
        values = np.linspace(pl - self.iota * span_p, ph + self.iota * span_p, self.Np)
        return {idx: float(val) for idx, val in enumerate(values)}
    
    def get_discrete_states(self):
        self.v_discrete = self.get_grid_point_values_v()
        self.x_discrete = self.get_grid_point_values_x()
        self.p_discrete = self.get_grid_point_values_p()

    def continuous_to_discrete(self, p, v):
        p_idx = min(self.p_discrete, key=lambda x: abs(self.p_discrete[x] - p))
        v_idx = min(self.v_discrete, key=lambda x: abs(self.v_discrete[x] - v))
        return p_idx, v_idx

    def initialize_Q(self):
        for p in range(self.Np):
            for v in range(self.Nv):
                state = (p, v)
                for x in range(self.n_actions):
                    value = 0
                    for x_i in range(self.n_actions):
                        value += self.v_discrete[v] - (self.v_bar + self.lambdaN * (self.x_n + (self.I - 1) * self.x_discrete[x_i]))
                    value *= self.x_discrete[x] / ((1 - self.rho) * self.n_actions)
                    self.Q.update(state, x, value)

    def check_convergence(self, state, action):
        # print(self.last_optimal)
        if action != self.last_optimal.get(state, None):
            self.last_optimal[state] = action
            self.convergence_counter = 0

        else:
            self.convergence_counter += 1


class PreferredHabitatAgent:

    def __init__(self, config):
        self.xi = config.xi
        self.v_bar = config.v_bar

    def get_action(self, pt):
        z = -self.xi * (pt - self.v_bar)
        return z
    
class CircularBuffer:
    """
    Circular buffer for storing historical data.
    """
    def __init__(self, size):
        self.size = size
        self.buffer = np.ones(size)
        self.index = 0

    def add(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size

    def get(self):
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))

class AdaptiveMarketMaker:

    def __init__(self, config):
        self.theta = config.theta
        self.Tm = config.Tm

        self.vars_ = ['v','p','z','y']
        self.historical_data = {var: CircularBuffer(size = self.Tm) for var in self.vars_}

    def OLS(self, y, X):
        """
        Perform Ordinary Least Squares (OLS) regression.
        Parameters:
        y (CircularBuffer): The dependent variable.
        X (CircularBuffer): The independent variable(s).
        Returns:
        coef_ (ndarray): The estimated coefficients for the linear regression model.
        """
        y = y.get()
        X = X.get()
        
        X = np.vstack([X, np.ones(len(X))]).T
        coef_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coef_
    
    def determine_price(self, yt):
        """
        Determines the price based on historical data and a given input.
        This method uses Ordinary Least Squares (OLS) regression to calculate
        yt (float): The input value for which the price needs to be determined.
        Returns:
        float: The determined price based on the input `yt`.
        """

        xi_1, _ = self.OLS(self.historical_data['z'], self.historical_data['p'])
        gamma_1, gamma_0 = self.OLS(self.historical_data['v'], self.historical_data['y'])
        lambda_ = (xi_1 + self.theta * gamma_1) / (xi_1**2 + self.theta)
        # print(lambda_)
        price = gamma_0 + lambda_ * yt
        return price
    
    def update(self, vt, pt, zt, yt):
        """
        Updates the historical data with the given values.
        Parameters:
        vt (float): The value of `v` at time `t`.
        pt (float): The value of `p` at time `t`.
        zt (float): The value of `z` at time `t`.
        yt (float): The value of `y` at time `t`.
        """
        for var, value in zip(self.vars_, [vt, pt, zt, yt]):
            self.historical_data[var].add(value)

class NoiseAgent:
    def __init__(self, config):
        self.sigma = config.sigma_u

    def get_action(self):
        return np.random.normal(scale = self.sigma)

class Q_table_Batch:

    def __init__(self,
                 config,
                 B = 1000):
        self.Np = range(config.Np)
        self.Nv = range(config.Nv)
        self.B = B
        self.I = config.I
        self.states = list(product(self.Np, self.Nv))

        # self.Q = {s:np.zeros(config.Nx) for s in self.states}
        self.Q = torch.zeros((B,config.I, config.Np, config.Nv, config.Nx), dtype = torch.float32)
        
    def get_Q_value(self, state, action):
        # action:
        p, v = state[:, 0], state[:, 1] # Bx1, Bx1
        b_idx = torch.arange(self.B).unsqueeze(1).expand(self.B, self.I)        # (B, I)
        i_idx = torch.arange(self.I).unsqueeze(0).expand(self.B, self.I)        # (B, I)
        p_idx = p.unsqueeze(1).expand(self.B, self.I)                      # (B, I)
        v_idx = v.unsqueeze(1).expand(self.B, self.I)                  # (B, I)
        return self.Q[b_idx, i_idx, p_idx, v_idx, action]

    def get_best_action(self, state):
        p, v = state[:, 0], state[:, 1] # Bx1, Bx1
        return torch.argmax(self.Q[torch.arange(self.B),:, p, v], axis = -1) # B x I

    def get_best_value(self, state):
        p, v = state[:, 0], state[:, 1] # Bx1, Bx1
        return self.Q[torch.arange(self.B),:, p, v].max(axis = -1).values # B x I

    def update(self, state, action, value):
        p, v = state[:, 0], state[:, 1] # Bx1, Bx1
        b_idx = torch.arange(self.B).unsqueeze(1).expand(self.B, self.I)        # (B, I)
        i_idx = torch.arange(self.I).unsqueeze(0).expand(self.B, self.I)        # (B, I)
        p_idx = p.unsqueeze(1).expand(self.B, self.I)                      # (B, I)
        v_idx = v.unsqueeze(1).expand(self.B, self.I)                  # (B, I)
        self.Q[b_idx, i_idx, p_idx, v_idx, action] = value

class InformedAgents_Batch:

    def __init__(self, config, B = 1000):
        
        # parameters
        self.n_actions = config.Nx
        self.Np = config.Np
        self.Nv = config.Nv
        self.n_states = self.Np * self.Nv

        self.rho = config.rho
        self.alpha = config.alpha
        self.beta = config.beta

        self.sigma_v = config.sigma_v
        self.v_bar = config.v_bar
        self.sigma_u = config.sigma_u
        self.xi = config.xi
        self.theta = config.theta
        self.iota = config.iota
        self.I = config.I
        self.B = B
        # Q-table for RL
        self.Q = Q_table_Batch(config, B = B)

        # state count dictionary for epsilon decay
        # self.state_count = defaultdict(int)
        self.state_count = torch.zeros((B, self.I, self.Np, self.Nv), dtype = torch.long)

        # convergence dictionary
        self.last_optimal = torch.zeros((B, self.I, self.Np, self.Nv), dtype = torch.long)
        self.convergence_counter = torch.zeros((B, self.I), dtype = torch.long)

        # discretization of states
        self.get_discrete_states()

        # initialize Q-table
        self.initialize_Q()

    
    def get_epsilon(self, state):
        p, v = state[:, 0], state[:, 1] # Bx1, Bx1
        count = self.state_count[torch.arange(self.B), :, p, v] # B x I
        # print(count.shape)
        # print(count)
        self.state_count[torch.arange(self.B),:, p, v] += 1
        return np.exp(-self.beta * count) # B x I
    
    def get_action(self, state):
        epsilon = self.get_epsilon(state) # B x I
        # print(epsilon.shape)
        greedy = torch.randn(self.B, self.I) > epsilon
        # print(greedy)
        optimal_action = self.Q.get_best_action(state) # B x I

        self.check_convergence(state, optimal_action)
        return torch.where(greedy, optimal_action, torch.randint(0, self.n_actions, (self.B, self.I)))
        # torch.randint()
        # if torch.randn(self.B, self.I) < epsilon:
        #     return torch.randint(self.n_actions, (self.B, self.I))
        #     return np.random.randint(self.n_actions)
        # else:
        #     optimal_action = self.Q.get_best_action(state)
        #     self.check_convergence(state, optimal_action)
        #     return optimal_action
        
    def update(self, state, action, reward, next_state):
        learning = self.alpha * (reward + self.rho * self.Q.get_best_value(next_state)) # B x I
        memory = (1 - self.alpha) * self.Q.get_Q_value(state, action) # B x I
        value = learning + memory # B x I
        self.Q.update(state, action, value) 
    
    def get_grid_point_values_v(self):
        """
        Returns a zero indexed dictionary of the grid points for the state space of v
        """
        standard_normal = torch.distributions.Normal(0, 1)
        grid_point = [(2 * k - 1) / (2 * self.Nv) for k in range(1, self.Nv + 1)]
        values = standard_normal.icdf(torch.tensor(grid_point))
        return torch.tensor([self.v_bar + self.sigma_v * value for value in values], dtype = torch.float32)
    
    def get_grid_point_values_x(self):
        self.chiN, self.lambdaN = solve_chiN(I = self.I, xi = self.xi, sigma_u = self.sigma_u, sigma_v = self.sigma_v, theta = self.theta)
        self.chiM, self.lambdaM = solve_chiM(I = self.I, xi = self.xi, sigma_u = self.sigma_u, sigma_v = self.sigma_v, theta = self.theta)
        self.x_n, self.x_m = self.chiN, self.chiM # assuming v - v_bar = 1
        span_x = abs(self.x_n - self.x_m)
        low, high = -max(self.x_n, self.x_m) - self.iota * span_x, max(self.x_n, self.x_m) + self.iota * span_x
        values = np.linspace(low, high, self.n_actions)
        values = torch.tensor(values, dtype = torch.float32)
        # return {idx: float(val) for idx, val in enumerate(values)}
        return values
    
    def get_grid_point_values_p(self):
        try:
            lambda_for_p = max(self.lambdaN, self.lambdaM)
        except:
            raise ValueError("Call get_grid_point_values_x() first")
        ph = self.v_bar + lambda_for_p * (self.I * max(self.x_n, self.x_m) + self.sigma_u * 1.96)
        pl = self.v_bar - lambda_for_p * (self.I * max(self.x_n, self.x_m) + self.sigma_u * 1.96)
        span_p = ph - pl
        values = np.linspace(pl - self.iota * span_p, ph + self.iota * span_p, self.Np)
        return torch.tensor(values, dtype = torch.float32)
    
    def get_discrete_states(self):
        self.v_discrete = self.get_grid_point_values_v()
        self.x_discrete = self.get_grid_point_values_x()
        self.p_discrete = self.get_grid_point_values_p()

    # def continuous_to_discrete(self, p, v):
    #     p_idx = min(self.p_discrete, key=lambda x: abs(self.p_discrete[x] - p))
    #     v_idx = min(self.v_discrete, key=lambda x: abs(self.v_discrete[x] - v))
    #     return p_idx, v_idx
    def continuous_to_discrete(self, p_values, v_values):
            # p_values and v_values are 1D tensors of continuous values.
            # For each value, we pick the index of the closest grid point.
            p_idx = (torch.abs(self.p_discrete.unsqueeze(0) - p_values.unsqueeze(1))).argmin(dim=1)
            # v_idx = (torch.abs(self.v_discrete.unsqueeze(0) - v_values.unsqueeze(1))).argmin(dim=1)
            v_idx = v_values
            return torch.stack((p_idx, v_idx), axis = 1) # B x 2

    def initialize_Q(self):
        for p in range(self.Np):
            for v in range(self.Nv):
                p_tensor = torch.zeros((self.B), dtype = torch.long) + p
                # print(p_tensor.dtype)
                v_tensor = torch.zeros((self.B), dtype = torch.long) + v
                state = torch.stack((p_tensor, v_tensor), dim = 1) # B x 2
                
                for x in range(self.n_actions):
                    value = 0
                    for x_i in range(self.n_actions):
                        value += self.v_discrete[v] - (self.v_bar + self.lambdaN * (self.x_n + (self.I - 1) * self.x_discrete[x_i]))
                    value *= self.x_discrete[x] / ((1 - self.rho) * self.n_actions)
                    self.Q.update(state, x, value)

    def check_convergence(self, state, action):
        # print(self.last_optimal)
        p, v = state[:, 0], state[:, 1] # Bx1, Bx1

        self.convergence_counter = torch.where(action != self.last_optimal[torch.arange(self.B),:, p, v], 
                                               0, 
                                               self.convergence_counter + 1)

class CircularBuffer_Batch:
    """
    Circular buffer for storing historical data.
    """
    def __init__(self, size, B = 1000):
        self.size = size
        self.buffer = torch.zeros((B, size), dtype = torch.float32)
        self.index = 0

    def add(self, value):
        self.buffer[:, self.index] = value # B x 1
        self.index = (self.index + 1) % self.size

    def get(self):
        return torch.cat((self.buffer[:, self.index:], self.buffer[:, :self.index]), dim = 1) # B x size
        # return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))

class AdaptiveMarketMaker_Batch:

    def __init__(self, config, B = 1000):
        self.theta = config.theta
        self.Tm = config.Tm

        self.vars_ = ['v','p','z','y']
        self.historical_data = {var: CircularBuffer_Batch(size = self.Tm, B = B) for var in self.vars_}
    def OLS(self, y, X):
        """
        Perform batched Ordinary Least Squares (OLS) regression using PyTorch.
        Assumes X and y are both (B x Tm).
        """
        y = y.get()  # B x Tm
        X = X.get()  # B x Tm

        B, _ = X.shape
        coef_batch = torch.zeros((2, B), dtype = torch.float32)  # 2 x B for slope and intercept
        for b in range(B):
            _X = np.vstack([X[b], np.ones(len(X[b]))]).T

            coef_, _, _, _ = np.linalg.lstsq(_X, y[b], rcond=None)
            coef_batch[:, b] = torch.tensor(coef_, dtype=torch.float32)
        # print(coef_batch.shape)
        # print(coef_batch)
        return coef_batch # 2 x B

        # B, T = X.shape

        # # Add intercept: (B x T x 2) where last dim is [x_t, 1]
        # X_aug = torch.stack([X, torch.ones_like(X)], dim=2)  # B x T x 2

        # # Reshape y to (B x T x 1)
        # y = y.unsqueeze(-1)  # B x T x 1

        # # Batched least squares using pseudo-inverse: beta = (X^T X)^-1 X^T y
        # X_T = X_aug.transpose(1, 2)  # B x 2 x T
        # beta = torch.linalg.pinv(X_T @ X_aug) @ X_T @ y  # B x 2 x 1

        # return beta.squeeze(-1)  # B x 2

    
    def determine_price(self, yt):
        """
        Determines the price based on historical data and a given input.
        This method uses Ordinary Least Squares (OLS) regression to calculate
        yt (float): The input value for which the price needs to be determined.
        Returns:
        float: The determined price based on the input `yt`.
        """

        xi_1, _ = self.OLS(self.historical_data['z'], self.historical_data['p']) # B, _
        # print(xi_1.shape)
        gamma_1, gamma_0 = self.OLS(self.historical_data['v'], self.historical_data['y']) # B, 2
        lambda_ = (xi_1 + self.theta * gamma_1) / (xi_1**2 + self.theta) # B, 1
        # print(lambda_)
        price = gamma_0 + lambda_ * yt # B, 1 + B, 1 * scalar = B, 1
        return price

    def update(self, vt, pt, zt, yt):
        """
        Updates the historical data with the given batched values.
        Parameters:
        vt (torch.Tensor): Batched values of `v` at time `t` with shape (B,).
        pt (torch.Tensor): Batched values of `p` at time `t` with shape (B,).
        zt (torch.Tensor): Batched values of `z` at time `t` with shape (B,).
        yt (torch.Tensor): Batched values of `y` at time `t` with shape (B,).
        """
        for var, value in zip(self.vars_, [vt, pt, zt, yt]):
            self.historical_data[var].add(value)