"""
Train A2C, PPO, DQN, or DDQN with the experimental protocol.

Protocol: 1,000,000 timesteps; periodic deterministic evaluation every 20,000
steps on 20 fixed episodes; final evaluation on 50 fixed episodes (seeds 2000-2049).

Examples
    # Main comparison: 7 seeds per algorithm
    python scripts/train.py --algo DQN --seeds 0 1 2 3 4 5 6 --outdir runs/main

    # Double DQN
    python scripts/train.py --algo DDQN --seeds 0 1 2 3 4 5 6 --outdir runs/ddqn

    # Reward sensitivity: 3 seeds per configuration
    python scripts/train.py --algo PPO --seeds 0 1 2 --reward-config no_rdelta --outdir runs/reward

    # Hyperparameter search (e.g., A2C with n_steps=80 and lr=1e-3)
    python scripts/train.py --algo A2C --seeds 0 1 2 --override '{"n_steps": 80, "learning_rate": 0.001}' --outdir runs/hparam
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch as th
from stable_baselines3.common.utils import set_random_seed

from gridnav import make_env, build_model, evaluate_model_detailed, TrainEvalCallback, REWARD_CONFIGS


def run(algo, seed, timesteps, reward_config, overrides, outdir, tag):
    set_random_seed(seed)  # network weights and learning algorithm
    env = make_env(reward_config, use_monitor=True)
    eval_env = make_env(reward_config)
    model = build_model(algo, env, overrides)
    callback = TrainEvalCallback(eval_env=eval_env, eval_freq=20_000, n_eval_episodes=20)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = time.time() - t0

    metrics = evaluate_model_detailed(make_env(reward_config), model, n_episodes=50)
    name = f"{tag}_{algo}_seed{seed}"
    model.save(os.path.join(outdir, f"model_{name}"))
    record = {"algorithm": algo, "seed": seed, "reward_config": reward_config,
              "overrides": overrides, "timesteps": timesteps,
              "wall_time_min": elapsed / 60, "steps_per_sec": timesteps / elapsed, **metrics}
    with open(os.path.join(outdir, f"metrics_{name}.json"), "w") as f:
        json.dump(record, f, indent=1)
    with open(os.path.join(outdir, f"curves_{name}.json"), "w") as f:
        json.dump(callback.history(), f)
    print(f"[{name}] SR={metrics['success_rate']:.3f}  MR={metrics['mean_reward']:.3f}  "
          f"steps={metrics['mean_steps']:.1f}  ({elapsed / 60:.1f} min)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", required=True, choices=["A2C", "PPO", "DQN", "DDQN"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--reward-config", default="baseline", choices=list(REWARD_CONFIGS))
    p.add_argument("--override", default="{}", help="JSON dict of hyperparameters overriding the defaults")
    p.add_argument("--outdir", default="runs")
    p.add_argument("--tag", default=None, help="prefix for output files (default: reward config name)")
    p.add_argument("--threads", type=int, default=1, help="torch threads per process")
    a = p.parse_args()

    th.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)
    overrides = json.loads(a.override)
    for seed in a.seeds:
        run(a.algo, seed, a.timesteps, a.reward_config, overrides, a.outdir, a.tag or a.reward_config)


if __name__ == "__main__":
    main()
