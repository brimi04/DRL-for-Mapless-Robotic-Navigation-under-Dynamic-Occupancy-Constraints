"""Evaluation utilities and the training callback used in all experiments."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Evaluation episode seeds (shared by all runs).
FINAL_EVAL_SEED_START = 2000      # final evaluation: episodes 2000..2049
PERIODIC_EVAL_SEED_START = 12345  # periodic evaluation: episodes 12345..12364


def evaluate_model_detailed(env, model, n_episodes=50, eval_seed_start=FINAL_EVAL_SEED_START):
    """
    Evaluate a model with a deterministic policy over `n_episodes` seeded episodes.

    Returns a dict with:
      success_rate, failure_rate, dynamic_collision_rate, truncation_rate,
      mean_reward, std_reward, mean_steps, std_steps,
      mean_steps_success, std_steps_success, mean_steps_failure, std_steps_failure,
      block_events_total, mean_block_events, std_block_events,
      mean_dyn_block_steps, std_dyn_block_steps
    Block events count forward actions blocked by a dynamic obstacle.
    """
    successes = dynamic_collisions = truncations = 0
    rewards, steps_list, steps_success, steps_failure, block_events = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=eval_seed_start + ep)
        done, total_reward, steps, ep_blocks, last_info = False, 0.0, 0, 0, {}
        prev_pos = env.agent_pos

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            total_reward += reward
            steps += 1

            # Forward action that did not move the agent because the target cell
            # was occupied by a dynamic obstacle.
            if action == 1 and env.agent_pos == prev_pos:
                move = env.DIRS[env.heading]
                new_pos = (env.agent_pos[0] + int(move[0]), env.agent_pos[1] + int(move[1]))
                if new_pos in env.dynamic_obstacles:
                    ep_blocks += 1

            prev_pos = env.agent_pos
            done = terminated or truncated

        rewards.append(total_reward)
        steps_list.append(steps)
        block_events.append(ep_blocks)

        if env.agent_pos == env.goal_pos:
            successes += 1
            steps_success.append(steps)
        elif last_info.get("collision_dynamic", False):
            dynamic_collisions += 1
            steps_failure.append(steps)
        else:
            truncations += 1
            steps_failure.append(steps)

    def _mean(x):
        return float(np.mean(x)) if x else float("nan")

    def _std(x):
        return float(np.std(x)) if x else float("nan")

    # Proxy for time between consecutive blockages (episodes with at least one blockage)
    dwell = [s / (b + 1) for s, b in zip(steps_list, block_events) if b > 0]

    return {
        "success_rate": successes / n_episodes,
        "failure_rate": (dynamic_collisions + truncations) / n_episodes,
        "dynamic_collision_rate": dynamic_collisions / n_episodes,
        "truncation_rate": truncations / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "mean_steps_success": _mean(steps_success),
        "std_steps_success": _std(steps_success),
        "mean_steps_failure": _mean(steps_failure),
        "std_steps_failure": _std(steps_failure),
        "block_events_total": int(np.sum(block_events)),
        "mean_block_events": float(np.mean(block_events)),
        "std_block_events": float(np.std(block_events)),
        "mean_dyn_block_steps": _mean(dwell),
        "std_dyn_block_steps": _std(dwell),
    }


class TrainEvalCallback(BaseCallback):
    """Records training episodes and runs a periodic deterministic evaluation every `eval_freq` steps."""

    def __init__(self, eval_env, eval_freq=20_000, n_eval_episodes=20,
                 eval_seed_start=PERIODIC_EVAL_SEED_START, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed_start = eval_seed_start
        self._next_eval = eval_freq

        self.train_timesteps, self.train_episode_rewards, self.train_episode_lengths = [], [], []
        self.eval_timesteps, self.eval_mean_rewards = [], []
        self.eval_success_rates, self.eval_mean_steps = [], []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.train_timesteps.append(self.num_timesteps)
                self.train_episode_rewards.append(ep["r"])
                self.train_episode_lengths.append(ep["l"])

        if self.num_timesteps >= self._next_eval:
            m = evaluate_model_detailed(self.eval_env, self.model,
                                        n_episodes=self.n_eval_episodes,
                                        eval_seed_start=self.eval_seed_start)
            self.eval_timesteps.append(self.num_timesteps)
            self.eval_mean_rewards.append(m["mean_reward"])
            self.eval_success_rates.append(m["success_rate"])
            self.eval_mean_steps.append(m["mean_steps"])
            self._next_eval += self.eval_freq
        return True

    def history(self):
        return {
            "train_timesteps": self.train_timesteps,
            "train_episode_rewards": self.train_episode_rewards,
            "train_episode_lengths": self.train_episode_lengths,
            "eval_timesteps": self.eval_timesteps,
            "eval_mean_rewards": self.eval_mean_rewards,
            "eval_success_rates": self.eval_success_rates,
            "eval_mean_steps": self.eval_mean_steps,
        }
