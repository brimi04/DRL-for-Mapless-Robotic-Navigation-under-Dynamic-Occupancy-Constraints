"""The released models reproduce the released per-seed results (deterministic evaluation)."""
import os
import sys

import pandas as pd
import pytest
import torch as th

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from gridnav import make_env, evaluate_model_detailed  # noqa: E402
from gridnav.algorithms import load_model  # noqa: E402


@pytest.mark.parametrize("algo,seed,folder,csv", [
    ("A2C", 0, "main", "main"), ("PPO", 3, "main", "main"),
    ("DQN", 6, "main", "main"), ("DDQN", 2, "ddqn", "ddqn"),
])
def test_released_model_matches_results(algo, seed, folder, csv):
    th.set_num_threads(1)
    model = load_model(algo, os.path.join(ROOT, "models", folder, f"model_{algo}_seed{seed}.zip"))
    m = evaluate_model_detailed(make_env(), model, n_episodes=50)
    ref = pd.read_csv(os.path.join(ROOT, "results", csv, "per_seed_metrics.csv"))
    row = ref[(ref.algorithm == algo) & (ref.seed == seed)].iloc[0]
    assert m["success_rate"] == pytest.approx(row.success_rate)
    assert m["mean_reward"] == pytest.approx(row.mean_reward)
    assert m["mean_steps"] == pytest.approx(row.mean_steps)
