"""Experiment 4: Truthfulness vs informational substitutes (Theorems 1-3).

A focal agent considers deviating from truthful reporting by scaling its signal:
it reports log-odds ``logit(prior) + gamma * own_llr`` (``gamma = 1`` is truthful,
``gamma > 1`` is over-confident). Its reward is the cross-entropy against a reference
agent that observes the focal report *and* has ``k`` independent signals.

Prediction: with few substitutes the focal agent profits by over-reporting; as ``k``
grows the reference becomes a good proxy for the ground truth and the best response
converges to truthful reporting (``gamma -> 1``).
"""
from __future__ import annotations

import numpy as np

from _common import save_fig, write_csv
import matplotlib.pyplot as plt

from srpm import binary_symmetric, expected_payoff_vs_gamma, optimal_gamma

PRIOR = 0.5
FOCAL_Q = 0.7
REF_Q = 0.65          # quality of each of the reference's own signals
K_VALUES = [0, 2, 8, 32, 64, 128, 256]
GAMMAS = np.linspace(0.0, 3.0, 61)
N_SAMPLES = 400_000


def main() -> None:
    rng = np.random.default_rng(7)
    focal = binary_symmetric(FOCAL_Q)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    rows, k_axis, best_gammas = [], [], []

    for k in K_VALUES:
        ref = [binary_symmetric(REF_Q) for _ in range(k)]
        payoffs = expected_payoff_vs_gamma(
            PRIOR, focal, ref, GAMMAS, n_samples=N_SAMPLES, rng=rng
        )
        # Normalise each curve to its truthful (gamma=1) value for comparison.
        truthful_idx = int(np.argmin(np.abs(GAMMAS - 1.0)))
        gain = payoffs - payoffs[truthful_idx]
        ax1.plot(GAMMAS, gain, label=f"k={k}")
        g_star = optimal_gamma(GAMMAS, payoffs)
        k_axis.append(k)
        best_gammas.append(g_star)
        for g, p in zip(GAMMAS, payoffs):
            rows.append([k, round(float(g), 3), float(p)])

    ax1.axvline(1.0, ls="--", color="gray", lw=1)
    ax1.axhline(0.0, ls=":", color="gray", lw=1)
    ax1.set(xlabel="deviation gamma (1 = truthful)",
            ylabel="expected payoff gain vs truthful",
            title="Gain from deviating shrinks as k grows")
    ax1.legend(title="ref. substitutes", fontsize=8)
    ax1.grid(alpha=0.3)

    xpos = np.arange(len(k_axis))
    ax2.plot(xpos, best_gammas, marker="o", color="tab:red")
    ax2.axhline(1.0, ls="--", color="gray", label="truthful")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([str(k) for k in k_axis])
    ax2.set(xlabel="number of reference informational substitutes k",
            ylabel="profit-maximising gamma",
            title="Best response converges to truthful")
    ax2.legend()
    ax2.grid(alpha=0.3)

    save_fig(fig, "exp_incentives.png")
    write_csv("exp_incentives.csv", ["k", "gamma", "expected_payoff"], rows)
    print("Profit-maximising gamma by k (1.0 == truthful is optimal):")
    for k, g in zip(k_axis, best_gammas):
        print(f"  k={k:3d} -> gamma*={g:.2f}")


if __name__ == "__main__":
    main()
