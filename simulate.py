# simulate.py  (put next to agents.py etc.)
import argparse, json, numpy as np
from pathlib import Path
from util import simulate
from config import Config

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="path to the config file")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", required=True,
                   help="where to write the .npy log")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--constrained", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse()
    if args.seed is not None:
        np.random.seed(args.seed)
    cfg = Config.load(args.config)          # implement .load() however you like
    # fn = simulate_constrained if args.constrained else simulate
    simulate(T=args.steps, config=cfg, save_path=args.out)
