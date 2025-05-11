from util import simulate_batch
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

c = Config(sigma_u = 0.1)

if args.sigma_u is not None:
    c.sigma_u = args.sigma_u
print(f"Running simulation with σᵤ = {c.sigma_u}")
counter = 0
convergence_threshold = 1000000
save = '/rds/general/user/mc4724/home/data/sigma_u_' + str(c.sigma_u)+'_part_0'+'.pt'
# save = 'data_0.pt'
log, agents = simulate_batch(T = 50000, B=1000, config = c, 
                           save_path=save)
i = 1
while counter < convergence_threshold:
    new_save = '/rds/general/user/mc4724/home/data/sigma_u_' + str(c.sigma_u)+'_part_{0}'.format(i)+'.pt'
    # new_save = 'data_{0}.pt'.format(i)
    log, agents = simulate_batch(T = 50000, B=1000, config = c, 
                           save_path= new_save,
                           continue_simulation=save)
    i += 1
    save = new_save
    # Check if the agents have converged
    counter = log['convergence_counter']
    print(f"Convergence counter: {counter}")
    # if i > 3:
    #     break