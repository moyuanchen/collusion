from util import simulate
from config import Config


c = Config(sigma_u=1)
counter = 0
convergence_threshold = 1000000
while counter < convergence_threshold:
    log, agents = simulate(T = 500000, config = c, save_path='~/collusion/data/sigma_u_1e0.pkl')
    # Check if the agents have converged
    counter = min([agent.convergence_counter for agent in agents])

