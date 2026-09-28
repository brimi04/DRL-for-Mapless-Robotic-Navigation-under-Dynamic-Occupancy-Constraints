"""
RobotGridWorldEnv: mapless GridWorld navigation with mobile occupancy constraints.

This is the environment used in all experiments. The dynamics and
the reward are identical to the original experimental code; only comments,
docstrings, and the optional reward coefficients (used by the reward-sensitivity
study) were added.

Actions
    0 = turn left   (rotate 90 degrees, no displacement)
    1 = forward     (move one cell in the current heading)
    2 = turn right  (rotate 90 degrees, no displacement)

Observation (11 dimensions)
    [d_front, d_left, d_right, d_front_left, d_front_right,
     dx, dy, h_N, h_E, h_S, h_W]
    d_*     : free distance until the first blocked cell, normalized to [0, 1]
    dx, dy  : agent-to-goal displacement in the global grid frame, in [-1, 1]
    h_*     : one-hot encoding of the current heading (N/E/S/W)

Dynamic obstacles
    Modeled as mobile occupancy constraints. At each step, the five options of
    each obstacle (stay, N, E, S, W) are visited in random order and the first
    valid one is executed. They block cells but never enter the agent's cell,
    so no terminal collision occurs; failures happen only by truncation.
"""
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


# Reward configurations of the reward-sensitivity study.
# rf_coef: frontal free-space term, rdelta_coef: progress term, rp_val: step penalty.
REWARD_CONFIGS = {
    "baseline":    {"rf_coef": 0.03, "rdelta_coef": 0.25, "rp_val": -0.02},
    "no_rf":       {"rf_coef": 0.00, "rdelta_coef": 0.25, "rp_val": -0.02},
    "no_rdelta":   {"rf_coef": 0.03, "rdelta_coef": 0.00, "rp_val": -0.02},
    "low_rdelta":  {"rf_coef": 0.03, "rdelta_coef": 0.10, "rp_val": -0.02},
    "high_rdelta": {"rf_coef": 0.03, "rdelta_coef": 0.50, "rp_val": -0.02},
    "no_rp":       {"rf_coef": 0.03, "rdelta_coef": 0.25, "rp_val":  0.00},
}


class RobotGridWorldEnv(gym.Env):
    """GridWorld for mapless robotic navigation (Gymnasium interface)."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Headings: 0 = N, 1 = E, 2 = S, 3 = W
    DIRS = {
        0: np.array([-1, 0]),
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
    }

    def __init__(
        self,
        grid_size: int = 20,
        n_obstacles: int = 50,
        n_dynamic_obstacles: int = 10,
        max_steps: int = 500,
        obstacle_seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        rf_coef: float = 0.03,
        rdelta_coef: float = 0.25,
        rp_val: float = -0.02,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.obstacle_seed = obstacle_seed  # kept for compatibility; not used

        # Reward coefficients (defaults = baseline configuration)
        self.rf_coef = rf_coef
        self.rdelta_coef = rdelta_coef
        self.rp_val = rp_val

        self.action_space = spaces.Discrete(3)
        low = np.array([0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.agent_pos = None
        self.goal_pos = None
        self.heading = None
        self.obstacles = set()
        self.dynamic_obstacles = set()
        self.steps = 0
        self.fig = self.ax = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _rng(self, seed=None):
        return np.random.default_rng(seed)

    def _random_empty_cell(self, rng, forbidden=None):
        forbidden = forbidden or set()
        while True:
            pos = (int(rng.integers(0, self.grid_size)),
                   int(rng.integers(0, self.grid_size)))
            if pos not in forbidden:
                return pos

    def _place_obstacles(self, rng):
        self.obstacles = set()
        forbidden = {self.agent_pos, self.goal_pos}
        while len(self.obstacles) < self.n_obstacles:
            pos = self._random_empty_cell(rng, forbidden=forbidden | self.obstacles)
            self.obstacles.add(pos)

    def _place_dynamic_obstacles(self, rng):
        self.dynamic_obstacles = set()
        forbidden = {self.agent_pos, self.goal_pos} | self.obstacles
        while len(self.dynamic_obstacles) < self.n_dynamic_obstacles:
            pos = self._random_empty_cell(rng, forbidden=forbidden | self.dynamic_obstacles)
            self.dynamic_obstacles.add(pos)

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _is_blocked(self, pos):
        return (not self._in_bounds(pos)
                or pos in self.obstacles
                or pos in self.dynamic_obstacles)

    def _distance_until_blocked(self, heading):
        """Normalized number of free cells ahead in a cardinal direction."""
        direction = self.DIRS[heading]
        pos = np.array(self.agent_pos, dtype=int)
        dist = 0
        while True:
            pos = pos + direction
            if self._is_blocked((int(pos[0]), int(pos[1]))):
                break
            dist += 1
        return dist / (self.grid_size - 1)

    def _distance_until_blocked_custom(self, direction_vec):
        """Normalized number of free cells ahead along an arbitrary (diagonal) direction."""
        pos = np.array(self.agent_pos, dtype=int)
        dist = 0
        while True:
            pos = pos + direction_vec
            p = (int(pos[0]), int(pos[1]))
            if not self._in_bounds(p) or p in self.obstacles or p in self.dynamic_obstacles:
                break
            dist += 1
        return dist / (self.grid_size - 1)

    def _get_obs(self):
        left_h = (self.heading - 1) % 4
        right_h = (self.heading + 1) % 4

        front = self._distance_until_blocked(self.heading)
        left = self._distance_until_blocked(left_h)
        right = self._distance_until_blocked(right_h)
        front_left = self._distance_until_blocked_custom(self.DIRS[self.heading] + self.DIRS[left_h])
        front_right = self._distance_until_blocked_custom(self.DIRS[self.heading] + self.DIRS[right_h])

        dr = self.goal_pos[0] - self.agent_pos[0]
        dc = self.goal_pos[1] - self.agent_pos[1]
        goal_dx = np.clip(dr / (self.grid_size - 1), -1, 1)
        goal_dy = np.clip(dc / (self.grid_size - 1), -1, 1)

        heading_one_hot = np.zeros(4, dtype=np.float32)
        heading_one_hot[self.heading] = 1.0

        return np.array(
            [front, left, right, front_left, front_right, goal_dx, goal_dy, *heading_one_hot],
            dtype=np.float32,
        )

    def _distance_to_goal(self, pos):
        return abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

    def _load_fixed_map(self, fixed_map):
        self.agent_pos = tuple(fixed_map["agent_pos"])
        self.goal_pos = tuple(fixed_map["goal_pos"])
        self.heading = int(fixed_map.get("heading", 0))
        self.obstacles = set(tuple(p) for p in fixed_map.get("obstacles", []))
        self.dynamic_obstacles = set(tuple(p) for p in fixed_map.get("dynamic_obstacles", []))

        for pos in [self.agent_pos, self.goal_pos]:
            if not self._in_bounds(pos):
                raise ValueError(f"Position {pos} is outside the grid.")
        if self.agent_pos in self.obstacles or self.agent_pos in self.dynamic_obstacles:
            raise ValueError("The agent start position coincides with an obstacle.")
        if self.goal_pos in self.obstacles or self.goal_pos in self.dynamic_obstacles:
            raise ValueError("The goal coincides with an obstacle.")

    def _move_dynamic_obstacles(self, rng):
        """Each obstacle tries {stay, N, S, W, E} in random order and takes the first valid option."""
        occupied = set(self.obstacles) | {self.agent_pos, self.goal_pos}
        new_positions = []
        for obs in self.dynamic_obstacles:
            candidates = [obs,
                          (obs[0] - 1, obs[1]), (obs[0] + 1, obs[1]),
                          (obs[0], obs[1] - 1), (obs[0], obs[1] + 1)]
            rng.shuffle(candidates)
            chosen = obs
            for p in candidates:
                if self._in_bounds(p) and p not in occupied and p not in new_positions:
                    chosen = p
                    break
            new_positions.append(chosen)
        self.dynamic_obstacles = set(new_positions)

    # ------------------------------------------------------------------ #
    # Gymnasium interface
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # NOTE: as in the original experiments, a reset without a seed draws a
        # new map from fresh OS entropy. Evaluation episodes are always reset
        # with explicit seeds and are therefore fully reproducible.
        self.rng = self._rng(seed)
        self.steps = 0
        options = options or {}

        if "fixed_map" in options:
            self._load_fixed_map(options["fixed_map"])
            return self._get_obs(), {"fixed_map": True}

        self.agent_pos = self._random_empty_cell(self.rng)
        self.goal_pos = self._random_empty_cell(self.rng, forbidden={self.agent_pos})
        self.heading = int(self.rng.integers(0, 4))
        self._place_obstacles(self.rng)
        self._place_dynamic_obstacles(self.rng)
        return self._get_obs(), {"fixed_map": False}

    def step(self, action):
        self.steps += 1
        self._move_dynamic_obstacles(self.rng)

        # Terminal dynamic collision. Never triggered: dynamic obstacles cannot
        # enter the agent's cell (kept for compatibility with the original code).
        if self.agent_pos in self.dynamic_obstacles:
            return self._get_obs(), -2.0, True, False, {"collision_dynamic": True}

        old_dist = self._distance_to_goal(self.agent_pos)
        reward = self.rp_val                         # r_p: step penalty

        if action == 0:                              # turn left
            self.heading = (self.heading - 1) % 4
            reward += -0.005                         # r_g: turn penalty
        elif action == 2:                            # turn right
            self.heading = (self.heading + 1) % 4
            reward += -0.005                         # r_g: turn penalty
        elif action == 1:                            # forward
            move = self.DIRS[self.heading]
            new_pos = (self.agent_pos[0] + int(move[0]), self.agent_pos[1] + int(move[1]))
            if self._is_blocked(new_pos):
                reward += -1.5                       # r_b: blocked-move penalty
            else:
                self.agent_pos = new_pos

        new_dist = self._distance_to_goal(self.agent_pos)
        reward += self.rdelta_coef * (old_dist - new_dist)                    # r_delta_d: progress
        reward += self.rf_coef * self._distance_until_blocked(self.heading)  # r_f: frontal free space

        terminated = self.agent_pos == self.goal_pos
        if terminated:
            reward += 5.0                            # r_goal: terminal reward

        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {"distance_to_goal": new_dist}

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for r, c in self.obstacles:
            grid[r, c] = -1.0
        for r, c in self.dynamic_obstacles:
            grid[r, c] = -0.4
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        grid[gr, gc] = 0.5
        grid[ar, ac] = 1.0

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.clear()
        self.ax.imshow(grid, cmap="viridis", vmin=-1, vmax=1)
        d = self.DIRS[self.heading]
        self.ax.arrow(ac, ar, d[1] * .35, d[0] * .35, head_width=.2, head_length=.2, fc="red", ec="red")
        self.ax.set_title("Robot GridWorld")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        plt.show(block=False); plt.pause(.1)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = self.ax = None


def make_env(reward_config: str = "baseline", use_monitor: bool = False):
    """Build the training/evaluation environment (20x20, 50 static, 10 dynamic, 500 steps)."""
    from stable_baselines3.common.monitor import Monitor
    env = RobotGridWorldEnv(grid_size=20, n_obstacles=50, n_dynamic_obstacles=10,
                            max_steps=500, **REWARD_CONFIGS[reward_config])
    return Monitor(env) if use_monitor else env
