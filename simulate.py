from util import simulate
from config import Config
import argparse, json, numpy as np
from pathlib import Path

# parse command line arguments
parser = argparse.ArgumentParser(
    description="Run one simulation with an optional sigma_u override."
)
parser.add_argument("--sigma_u", type=float,
                    help="Override noise-trader σᵤ (float)")
args = parser.parse_args()

c = Config(sigma_u=100)

if args.sigma_u is not None:
    c.sigma_u = args.sigma_u

counter = 0
convergence_threshold = 1000000
while counter < convergence_threshold:
    log, agents = simulate(T = 500000, config = c, 
                           save_path='/rds/general/user/mc4724/home/data/sigma_u_' + str(c.sigma_u))
    # Check if the agents have converged
    counter = min([agent.convergence_counter for agent in agents['informed']])
