"""Basic checks of the environment."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnav import RobotGridWorldEnv, FIXED_MAPS, REWARD_CONFIGS  # noqa: E402


def test_observation_and_action_spaces():
    env = RobotGridWorldEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (11,)
    assert env.action_space.n == 3
    assert env.observation_space.contains(obs)


def test_seeded_reset_is_reproducible():
    a, b = RobotGridWorldEnv(), RobotGridWorldEnv()
    a.reset(seed=2000); b.reset(seed=2000)
    assert a.agent_pos == b.agent_pos and a.goal_pos == b.goal_pos
    assert a.obstacles == b.obstacles and a.dynamic_obstacles == b.dynamic_obstacles
    rng = np.random.default_rng(0)
    for _ in range(100):
        act = int(rng.integers(3))
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
        assert np.allclose(oa, ob) and ra == rb


def test_no_terminal_dynamic_collisions():
    env = RobotGridWorldEnv()
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(500):
        _, _, term, trunc, info = env.step(int(rng.integers(3)))
        assert not info.get("collision_dynamic", False)
        assert env.agent_pos not in env.dynamic_obstacles
        if term or trunc:
            break


def test_fixed_maps_occupancy():
    expected = {"map1": 40.5, "map2": 37.5, "map3": 59.5}
    for name, m in FIXED_MAPS.items():
        occ = 100 * (len(m["obstacles"]) + len(m["dynamic_obstacles"])) / 400
        assert abs(occ - expected[name]) < 1e-9
        env = RobotGridWorldEnv()
        obs, _ = env.reset(seed=0, options={"fixed_map": m})
        assert obs.shape == (11,)


def test_default_reward_is_baseline():
    env = RobotGridWorldEnv()
    cfg = REWARD_CONFIGS["baseline"]
    assert (env.rf_coef, env.rdelta_coef, env.rp_val) == (cfg["rf_coef"], cfg["rdelta_coef"], cfg["rp_val"])
