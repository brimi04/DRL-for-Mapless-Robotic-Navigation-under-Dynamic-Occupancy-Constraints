"""
Empirical stay probability of the dynamic obstacles.

At each step, the five options of each dynamic obstacle (stay, N, S, W, E) are
visited in random order and the first valid one is executed, so the stay
probability is 1/(k+1), where k is the number of locally valid moves. This
script measures the mean stay probability under the training configuration
(20x20 grid, 50 static and 10 dynamic obstacles) with a uniformly random agent (about 0.23; about 0.24 at the first step of
an episode, and similar with the trained policies).

Example
    python scripts/dynamic_obstacle_stats.py --episodes 200
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from gridnav import RobotGridWorldEnv


def run(n_episodes, max_steps, seed):
    env = RobotGridWorldEnv(grid_size=20, n_obstacles=50, n_dynamic_obstacles=10, max_steps=max_steps)
    action_rng = np.random.default_rng(seed)
    stays = total = 0
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            # Replicate the obstacle update of the environment step by step to
            # record, for each obstacle, whether "stay" was the executed option.
            occupied = set(env.obstacles) | {env.agent_pos, env.goal_pos}
            state = env.rng.bit_generator.state
            new_positions = []
            for obs in env.dynamic_obstacles:
                candidates = [obs, (obs[0] - 1, obs[1]), (obs[0] + 1, obs[1]),
                              (obs[0], obs[1] - 1), (obs[0], obs[1] + 1)]
                env.rng.shuffle(candidates)
                chosen = obs
                for p in candidates:
                    if env._in_bounds(p) and p not in occupied and p not in new_positions:
                        chosen = p
                        break
                new_positions.append(chosen)
                stays += int(chosen == obs)
                total += 1
            env.rng.bit_generator.state = state  # restore: the real step draws the same numbers
            _, _, terminated, truncated, _ = env.step(int(action_rng.integers(0, 3)))
            assert set(new_positions) == env.dynamic_obstacles  # replication is exact
            done = terminated or truncated
    return stays / total, total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    p_stay, n = run(a.episodes, a.max_steps, a.seed)
    print(f"Empirical stay probability: {p_stay:.3f}  ({n:,} obstacle updates)")


if __name__ == "__main__":
    main()
