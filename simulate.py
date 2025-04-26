from util import simulate
from config import Config


c = Config(sigma_u=10)
counter = 0
convergence_threshold = 1000000
while counter < convergence_threshold:
    log, agents = simulate(T = 50, config = c, save_path='./sigma_u_10')
    # Check if the agents have converged
    counter = min([agent.convergence_counter for agent in agents['informed']])
    break
