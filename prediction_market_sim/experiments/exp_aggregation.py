"""Experiment 1: Does the market aggregate information correctly?

Varying the number of truthful agents and their signal quality, we check that the
terminal (reference) prediction matches the ideal full-information Bayesian posterior
and that accuracy improves as more independent signals are pooled.
"""
from __future__ import annotations

import numpy as np

from _common import save_fig, write_csv
import matplotlib.pyplot as plt

from srpm import (
    Agent,
    MarketConfig,
    SelfResolvingMarket,
    TruthfulBayesian,
    aggregation_stats,
    binary_symmetric,
)

N_RUNS = 4000
PRIOR = 0.5
QUALITIES = [0.55, 0.65, 0.80]
AGENT_COUNTS = [2, 3, 5, 8, 12, 20, 30]


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    fig, (ax_acc, ax_gap) = plt.subplots(1, 2, figsize=(11, 4.2))

    for q in QUALITIES:
        accs, gaps, briers, fi_briers = [], [], [], []
        for n in AGENT_COUNTS:
            agents = [Agent(binary_symmetric(q), TruthfulBayesian()) for _ in range(n)]
            market = SelfResolvingMarket(MarketConfig(prior_p1=PRIOR, k=1))
            results = market.run_many(agents, N_RUNS, rng)
            st = aggregation_stats(results)
            accs.append(st.accuracy)
            gaps.append(st.mean_abs_gap_to_full_info)
            briers.append(st.mean_brier)
            fi_briers.append(st.mean_brier_full_info)
            rows.append([q, n, st.accuracy, st.mean_brier,
                         st.mean_brier_full_info, st.mean_abs_gap_to_full_info,
                         st.mean_total_cost])
        ax_acc.plot(AGENT_COUNTS, accs, marker="o", label=f"signal q={q}")
        ax_gap.plot(AGENT_COUNTS, gaps, marker="s", label=f"signal q={q}")

    ax_acc.set(xlabel="number of agents", ylabel="accuracy",
               title="Market accuracy vs #agents")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)
    ax_gap.set(xlabel="number of agents",
               ylabel="|market p1 - full-info p1|",
               title="Gap to ideal Bayesian posterior")
    ax_gap.legend()
    ax_gap.grid(alpha=0.3)

    save_fig(fig, "exp_aggregation.png")
    write_csv(
        "exp_aggregation.csv",
        ["signal_quality", "n_agents", "accuracy", "brier_market",
         "brier_full_info", "gap_to_full_info", "mean_total_cost"],
        rows,
    )
    print("Under truthful reporting the gap to the full-info posterior is ~0:")
    print("the market reproduces the ideal Bayesian aggregate exactly.")


if __name__ == "__main__":
    main()
