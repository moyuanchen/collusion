import numpy as np
import torch as torch
from collections import defaultdict
from utils import CircularBuffer, Q_table


class InformedAgent:

    def __init__(self, 
                 Np = 31,
                 Nv = 10,
                 Nx = 15, 
                 
                 rho = 0.95, 
                 alpha = 0.01, 
                 beta = 1e-5):
        
        self.n_actions = Nx
        self.n_states = Np * Nv

        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        self.Q = Q_table(Np = Np, Nv = Nv, Nx = Nx)

        self.state_count = defaultdict(int)
    
    def get_epsilon(self, state):
        v = self.state_count[state]
        self.state_count[state] += 1
        return np.exp(-self.beta * v)
    
    def get_action(self, state):
        epsilon = self.get_epsilon(state)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return self.Q.get_best_action(state)
        
    def update(self, state, action, reward, next_state):
        learning = self.alpha * (reward + self.rho * self.Q.get_best_value(next_state))
        memory = (1 - self.alpha) * self.Q.get_Q_value(state, action)
        value = learning + memory
        self.Q.update(state, action, value)

class PreferredHabitatAgent:

    def __init__(self, xi = 500, v_bar = 1):
        self.xi = xi
        self.v_bar = v_bar

    def get_action(self, pt):
        z = -self.xi * (pt - self.v_bar)
        return z

class AdaptiveMarketMaker:

    def __init__(self, theta, Tm):
        self.theta = theta
        self.Tm = Tm

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
        coefficients from historical data and then uses these coefficients to
        determine the price for a given input `yt`.
        Parameters:
        yt (float): The input value for which the price needs to be determined.
        Returns:
        float: The determined price based on the input `yt`.
        """

        xi_1, xi_0 = self.OLS(self.historical_data['z'], self.historical_data['p'])
        gamma_1, gamma_0 = self.OLS(self.historical_data['v'], self.historical_data['y'])
        lambda_ = (xi_1 + self.theta * gamma_1) / (xi_1**2 + self.theta)
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
    def __init__(self, sigma = 0.1):
        self.sigma = sigma

    def get_action(self):
        return np.random.normal(scale = self.sigma)