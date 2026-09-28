"""
Reproduce all statistics from the released per-seed results.

Outputs (printed):
  - Final performance (mean +/- sample std over seven seeds) and CV
  - Kruskal-Wallis tests (success rate, mean reward, mean steps, failure rate)
  - Pairwise Mann-Whitney U tests with Bonferroni correction and
    rank-biserial effect size r = 1 - 2U / (n1 n2)
  - Bootstrap 95% confidence intervals of the mean (10,000 resamples)
  - Convergence speed (first evaluation with SR >= 0.5 for three
    consecutive evaluations)
  - Levene test on the final periodic-evaluation reward
  - DDQN vs. DQN comparison
  - Reward sensitivity
  - Transfer to fixed maps
  - Hyperparameter search summary

Example
    python scripts/analyze.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
ALGOS = ["A2C", "PPO", "DQN"]


def fmt(x, s, d=3):
    return f"{x:.{d}f} +/- {s:.{d}f}"


def bootstrap_ci(values, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def convergence_step(timesteps, success_rates, threshold=0.5, window=3):
    for i in range(len(success_rates) - window + 1):
        if all(s >= threshold for s in success_rates[i:i + window]):
            return timesteps[i]
    return None


def main():
    main_df = pd.read_csv(os.path.join(ROOT, "main", "per_seed_metrics.csv"))
    curves = pd.read_csv(os.path.join(ROOT, "main", "learning_curves.csv"))
    by = {a: main_df[main_df.algorithm == a] for a in ALGOS}

    # ---------------- Final performance ----------------
    print("=" * 70 + "\nFINAL PERFORMANCE (7 seeds, mean +/- sample std)\n" + "=" * 70)
    for col, label, d in [("success_rate", "SR", 3), ("failure_rate", "FR/TR", 3),
                          ("mean_reward", "MR", 3), ("mean_steps", "MS", 1),
                          ("mean_steps_success", "MS_succ", 1)]:
        print(f"{label:8s} " + " | ".join(f"{a}: {fmt(by[a][col].mean(), by[a][col].std(), d)}" for a in ALGOS))
    cv, final_reward, conv = {}, {}, {}
    for a in ALGOS:
        mat = np.array([curves[(curves.algorithm == a) & (curves.seed == s)].sort_values("timestep").mean_reward.values
                        for s in range(7)])
        cv[a] = (mat.std(0) / (np.abs(mat.mean(0)) + 1e-8)).mean()
        final_reward[a] = mat[:, -1]
        conv[a] = []
        for s in range(7):
            c = curves[(curves.algorithm == a) & (curves.seed == s)].sort_values("timestep")
            t = convergence_step(c.timestep.tolist(), c.success_rate.tolist())
            conv[a].append(t if t is not None else 1_000_000)
    print("CV       " + " | ".join(f"{a}: {cv[a]:.3f}" for a in ALGOS))

    # ---------------- Kruskal-Wallis ----------------
    print("\nKruskal-Wallis")
    for col in ["success_rate", "mean_reward", "mean_steps", "failure_rate"]:
        H, p = stats.kruskal(*[by[a][col].values for a in ALGOS])
        print(f"  {col:14s} H = {H:.3f}, p = {p:.4f}")

    # ---------------- Pairwise comparisons ----------------
    print("\nPAIRWISE COMPARISONS - Mann-Whitney U (success rate), Bonferroni x3")
    for a1, a2 in [("A2C", "PPO"), ("A2C", "DQN"), ("PPO", "DQN")]:
        x, y = by[a1].success_rate.values, by[a2].success_rate.values
        U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        r = 1 - 2 * U / (len(x) * len(y))
        pc = min(3 * p, 1.0)
        print(f"  {a1} vs. {a2}: p_corr = {pc:.4f}, r = {r:.3f}, significant = {'Yes' if pc < 0.05 else 'No'}")

    # ---------------- Bootstrap CIs ----------------
    print("\nBootstrap 95% CI of the mean")
    for col in ["success_rate", "mean_reward", "mean_steps"]:
        print(f"  {col:14s} " + " | ".join(
            f"{a}: [{bootstrap_ci(by[a][col].values)[0]:.3f}, {bootstrap_ci(by[a][col].values)[1]:.3f}]" for a in ALGOS))

    # ---------------- Convergence and Levene ----------------
    print("\nCONVERGENCE SPEED (mean +/- population std; converged seeds)")
    for a in ALGOS:
        v = np.array(conv[a])
        print(f"  {a}: {v.mean():,.0f} +/- {v.std():,.0f}  ({(v < 1_000_000).sum()}/7)")
    W, p = stats.levene(*final_reward.values())
    print(f"\nLevene (final periodic-evaluation reward): W = {W:.3f}, p = {p:.3f}")

    # ---------------- DDQN ----------------
    dd = pd.read_csv(os.path.join(ROOT, "ddqn", "per_seed_metrics.csv"))
    print("\nDDQN (7 seeds): SR " + fmt(dd.success_rate.mean(), dd.success_rate.std())
          + ", MR " + fmt(dd.mean_reward.mean(), dd.mean_reward.std())
          + ", MS " + fmt(dd.mean_steps.mean(), dd.mean_steps.std(), 1))
    U, p = stats.mannwhitneyu(dd.success_rate.values, by["DQN"].success_rate.values, alternative="two-sided")
    print(f"  DDQN vs. DQN (success rate): U = {U:.1f}, p = {p:.3f}, r = {1 - 2 * U / 49:.3f}")

    # ---------------- Reward sensitivity ----------------
    rs = pd.read_csv(os.path.join(ROOT, "reward_sensitivity", "per_seed_success.csv"))
    print("\nREWARD SENSITIVITY (3 seeds, mean +/- sample std)")
    for cfg in ["baseline", "no_rf", "no_rdelta", "low_rdelta", "high_rdelta", "no_rp"]:
        sub = rs[rs.reward_config == cfg]
        print(f"  {cfg:12s} " + " | ".join(
            f"{a}: {fmt(sub[sub.algorithm == a].success_rate.mean(), sub[sub.algorithm == a].success_rate.std())}"
            for a in ALGOS))

    # ---------------- Transfer to fixed maps ----------------
    fm = pd.read_csv(os.path.join(ROOT, "fixed_maps", "per_seed_success.csv"))
    print("\nTRANSFER TO FIXED MAPS (7 seeds x 200 episodes; mean +/- sample std; seeds with SR > 0.5)")
    for m in ["map1", "map2", "map3"]:
        parts = []
        for a in ALGOS:
            v = fm[(fm["map"] == m) & (fm.algorithm == a)].success_rate
            parts.append(f"{a}: {fmt(v.mean(), v.std())} ({(v > 0.5).sum()}/7)")
        print(f"  {m}: " + " | ".join(parts))

    # ---------------- Hyperparameter search ----------------
    hp = pd.read_csv(os.path.join(ROOT, "hyperparameter_search", "per_seed_success.csv"))
    print("\nHyperparameter search (3 seeds): best configuration per algorithm")
    g = hp.groupby(["algorithm", "config"]).success_rate.agg(["mean", "std"]).reset_index()
    for a in ALGOS:
        best = g[g.algorithm == a].sort_values("mean", ascending=False).iloc[0]
        print(f"  {a}: {best['config']}  SR = {fmt(best['mean'], best['std'])}")


if __name__ == "__main__":
    main()
