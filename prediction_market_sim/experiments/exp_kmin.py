"""Experiment 5: Reproduce Figure 2 — minimum informational substitutes k_min.

Plots the analytic bound k_min (Theorems 1 & 5, Remark 3) as a function of the
prior market prediction ``y_1^{(t-1)}``, for the paper's three ``(δ, η)`` settings
and a sweep of target gaps ``ε``. This recreates Figure 2 in the paper, including
its headline takeaway: k_min is far more sensitive to ``δ`` than to ``η`` or ``ε``
(dropping δ from 1e-1 to 1e-3 inflates k_min ~100x; the y-axis jumps accordingly).

    cd prediction_market_sim
    python experiments/exp_kmin.py
"""
from __future__ import annotations

import numpy as np

from _common import save_fig, write_csv
import matplotlib.pyplot as plt

from srpm import k_min_curve

# The paper's three panels: (delta, eta).
PANELS = [
    (1e-1, 1e-1),
    (1e-1, 1e-3),
    (1e-3, 1e-1),
]
EPSILONS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
Y1_GRID = np.linspace(0.005, 0.995, 199)


def main() -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.4))
    rows = []
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(EPSILONS)))

    for ax, (delta, eta) in zip(axes, PANELS):
        for eps, c in zip(EPSILONS, colors):
            k = k_min_curve(delta, eta, eps, Y1_GRID)
            ax.plot(Y1_GRID, k, color=c, lw=1.6,
                    label=fr"$\epsilon = {eps:.0e}$")
            for y, kv in zip(Y1_GRID, k):
                rows.append([delta, eta, eps, round(float(y), 4), float(kv)])
        ax.set_title(fr"$\delta = {delta:.0e},\ \eta = {eta:.0e}$")
        ax.set_xlabel(r"prior market prediction $y_1^{(t-1)}$")
        ax.set_ylabel(r"$k_{min}$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    save_fig(fig, "exp_kmin.png")
    write_csv("exp_kmin.csv", ["delta", "eta", "epsilon", "y1", "k_min"], rows)

    # Console sanity table at the symmetric prior y1 = 0.5.
    print("\nk_min at y1 = 0.5 (rounded up to whole signals):")
    print(f"{'delta':>8}{'eta':>8}" + "".join(f"{f'e={e:.0e}':>12}" for e in EPSILONS))
    for delta, eta in PANELS:
        ks = [k_min_curve(delta, eta, e, np.array([0.5]))[0] for e in EPSILONS]
        print(f"{delta:>8.0e}{eta:>8.0e}"
              + "".join(f"{int(np.ceil(k)):>12d}" for k in ks))
    print("\nNote: k_min jumps ~100x between the delta=1e-1 and delta=1e-3 panels,\n"
          "reproducing the paper's 'delta matters far more than eta or epsilon'.")


if __name__ == "__main__":
    main()
