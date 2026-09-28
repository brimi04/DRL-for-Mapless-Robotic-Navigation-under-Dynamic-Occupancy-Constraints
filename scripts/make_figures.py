"""
Regenerate the result figures from the released per-seed results.

  figures/fig_boxplots.png                  final performance distribution (7 seeds)
  figures/fig_learning_curves_combined.pdf  learning curves: (a) success rate, (b) mean reward

Example
    python scripts/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results", "main")
FIGS = os.path.join(ROOT, "figures")
ALGOS = ["A2C", "PPO", "DQN"]
COLORS = {"A2C": "#4C72B0", "PPO": "#2CA02C", "DQN": "#D62728"}
LINESTYLES = {"A2C": (0, (1, 1.2)), "PPO": (0, (5, 2)), "DQN": "-"}  # dotted, dashed, solid


def boxplots(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Final Performance Distribution ( 7 seeds per algorithm )", fontsize=12)
    for ax, metric, label in zip(axes, ["success_rate", "mean_reward"], ["Success Rate", "Mean Reward"]):
        data = [df[df.algorithm == a][metric].values for a in ALGOS]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=ALGOS,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=5))
        for patch, a in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(COLORS[a]); patch.set_alpha(0.75)
        rng = np.random.default_rng(42)
        for i, (a, d) in enumerate(zip(ALGOS, data), start=1):
            ax.scatter(np.full(len(d), i) + rng.uniform(-0.08, 0.08, len(d)), d, color=COLORS[a],
                       s=30, zorder=5, edgecolors="white", linewidths=0.5)
        ax.set_ylabel(label, fontsize=10); ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig_boxplots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def learning_curves(curves):
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
    for ax, metric, label, title in [(axes[0], "success_rate", "Success rate", "(a) Success rate"),
                                     (axes[1], "mean_reward", "Mean reward", "(b) Mean reward")]:
        for a in ALGOS:
            sub = curves[curves.algorithm == a]
            mat = np.array([sub[sub.seed == s].sort_values("timestep")[metric].values for s in range(7)])
            ts = sub[sub.seed == 0].sort_values("timestep").timestep.values / 1e6
            m, s = mat.mean(0), mat.std(0)
            ax.fill_between(ts, m - s, m + s, color=COLORS[a], alpha=0.15, linewidth=0)
            ax.plot(ts, m, color=COLORS[a], linestyle=LINESTYLES[a], linewidth=2.0 if a == "A2C" else 1.8, label=a)
        ax.set_ylabel(label); ax.set_title(title, pad=3); ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, 1.0)
        ax.legend(loc="lower right", ncol=1, handlelength=2.8, labelspacing=0.25, borderpad=0.35,
                  frameon=True, framealpha=0.95, edgecolor="0.8")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel(r"Timesteps ($\times 10^6$)")
    plt.tight_layout(pad=0.3, h_pad=0.6)
    plt.savefig(os.path.join(FIGS, "fig_learning_curves_combined.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGS, "fig_learning_curves_combined.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    boxplots(pd.read_csv(os.path.join(RESULTS, "per_seed_metrics.csv")))
    learning_curves(pd.read_csv(os.path.join(RESULTS, "learning_curves.csv")))
    print(f"Figures saved in {os.path.abspath(FIGS)}")
