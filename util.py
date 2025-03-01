import numpy as np
from itertools import product

class Q_table:

    def __init__(self,
                 Np = 31,
                 Nv = 10, 
                 Nx = 15,
                 initial_values = None):

        self.Np = range(Np)
        self.Nv = range(Nv)

        states = list(product(self.Np, self.Nv))

        self.Q = {s:np.zeros(Nx) for s in states}

        # if initial_values is not None:
        #     self.Q = initial_values
        
    def get_Q_value(self, state, action):
        return self.Q[state][action]

    def get_best_action(self, state):
        return np.argmax(self.Q[state])

    def get_best_value(self, state):
        return np.max(self.Q[state])

    def update(self, state, action, value):
        self.Q[state][action] = value

class CircularBuffer:
    """
    Circular buffer for storing historical data.
    """
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size)
        self.index = 0

    def add(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size

    def get(self):
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))