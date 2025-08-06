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
parser.add_argument("--cont", type=str,
                    help="Continue simulation from a previous save (str)")
parser.add_argument("--out", type=str, default="/rds/general/user/mc4724/home/data",
                    help="Output directory for simulation data (str)")
args = parser.parse_args()
continue_simulation = args.cont
c = Config(sigma_u = 0.1)

if args.sigma_u is not None:
    c.sigma_u = args.sigma_u
print(f"Running simulation with σᵤ = {c.sigma_u}")
counter = 0
convergence_threshold = 1000000
# Define base directory for saving simulation data
base_path = Path(args.out)
# base_path = Path(".")
if continue_simulation:
    i = int(Path(continue_simulation).stem.split('_')[-1])
    i += 1
    save = base_path / f"sigma_u_{c.sigma_u}_part_{i}.pt"
    log, agents = simulate_batch(T=5000, B=100, config=c,
                                 save_path=str(save),
                                 continue_simulation=continue_simulation)
    
    # Remove older files if more than 10 exist
    # files = sorted(base_path.glob(f"sigma_u_{c.sigma_u}_part_*.pt"),
    #                key=lambda f: int(f.stem.split('_')[-1]))
    # if len(files) > 10:
    #     for old_file in files[:-10]:
    #         old_file.unlink()
else:
    save = base_path / f"sigma_u_{c.sigma_u}_part_0.pt"
    log, agents = simulate_batch(T=50, B=10, config=c,
                                 save_path=str(save))
    i = 1
    # Remove older files if more than 10 exist
    # files = sorted(base_path.glob(f"sigma_u_{c.sigma_u}_part_*.pt"),
    #                key=lambda f: int(f.stem.split('_')[-1]))
    # if len(files) > 10:
    #     for old_file in files[:-10]:
    #         old_file.unlink()

while counter < convergence_threshold:
    new_save = base_path / f"sigma_u_{c.sigma_u}_part_{i}.pt"
    log, agents = simulate_batch(T=50, B=10, config=c,
                                 save_path=str(new_save),
                                 continue_simulation=str(save))
    i += 1
    save = new_save
    counter = log['convergence_counter'].min()
    # print(log['convergence_counter'])
    # print(counter)
    print(f"Convergence counter: {counter}")
    
    # # Remove older files if more than 10 exist
    # files = sorted(base_path.glob(f"sigma_u_{c.sigma_u}_part_*.pt"),
    #                key=lambda f: int(f.stem.split('_')[-1]))
    # if len(files) > 10:
    #     for old_file in files[:-10]:
    #         old_file.unlink()
