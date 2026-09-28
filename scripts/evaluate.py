"""
Evaluate saved models on the 50 fixed evaluation episodes (seeds 2000-2049).

Evaluation is deterministic: the released models reproduce the per-seed
results in results/main/per_seed_metrics.csv and results/ddqn/per_seed_metrics.csv.

Examples
    python scripts/evaluate.py --models-dir models/main --algos A2C PPO DQN --out eval_main.csv
    python scripts/evaluate.py --models-dir models/ddqn --algos DDQN --out eval_ddqn.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch as th

from gridnav import make_env, evaluate_model_detailed
from gridnav.algorithms import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--algos", nargs="+", default=["A2C", "PPO", "DQN"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    p.add_argument("--out", default="evaluation.csv")
    a = p.parse_args()
    th.set_num_threads(1)

    rows = []
    for algo in a.algos:
        for seed in a.seeds:
            path = os.path.join(a.models_dir, f"model_{algo}_seed{seed}.zip")
            model = load_model(algo, path)
            m = evaluate_model_detailed(make_env(), model, n_episodes=50)
            rows.append({"algorithm": algo, "seed": seed, **m})
            print(f"{algo} seed={seed}: SR={m['success_rate']:.3f} MR={m['mean_reward']:.3f} "
                  f"steps={m['mean_steps']:.1f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
