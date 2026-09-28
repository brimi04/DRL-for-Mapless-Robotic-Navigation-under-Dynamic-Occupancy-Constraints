"""
Transfer evaluation on the three unseen structured maps.

Each model is evaluated for 200 episodes per map with a deterministic policy.
Start, goal, and static obstacles are fixed; the episode seed (0..199) only
changes the motion of the six dynamic obstacles.

Example
    python scripts/evaluate_fixed_maps.py --models-dir models/main --out fixed_maps.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch as th

from gridnav import RobotGridWorldEnv, FIXED_MAPS
from gridnav.algorithms import load_model


def success_rate_on_map(model, fixed_map, n_episodes=200):
    successes = 0
    for ep in range(n_episodes):
        env = RobotGridWorldEnv(grid_size=20, n_obstacles=50, n_dynamic_obstacles=10, max_steps=500)
        obs, _ = env.reset(seed=ep, options={"fixed_map": fixed_map})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        successes += int(env.agent_pos == env.goal_pos)
    return successes / n_episodes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--algos", nargs="+", default=["A2C", "PPO", "DQN"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    p.add_argument("--maps", nargs="+", default=list(FIXED_MAPS))
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default="fixed_maps.csv")
    a = p.parse_args()
    th.set_num_threads(1)

    rows = []
    for map_name in a.maps:
        for algo in a.algos:
            for seed in a.seeds:
                model = load_model(algo, os.path.join(a.models_dir, f"model_{algo}_seed{seed}.zip"))
                sr = success_rate_on_map(model, FIXED_MAPS[map_name], a.episodes)
                rows.append({"map": map_name, "algorithm": algo, "seed": seed, "success_rate": sr})
                print(f"{map_name} {algo} seed={seed}: SR={sr:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    summary = df.groupby(["map", "algorithm"])["success_rate"].agg(["mean", "std"])
    print(summary.round(3))


if __name__ == "__main__":
    main()
