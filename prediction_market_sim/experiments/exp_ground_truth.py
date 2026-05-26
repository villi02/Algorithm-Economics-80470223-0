"""Experiment 2: How much "ground truth" access does the market need?

We hold the population size fixed and vary how many agents have near-oracle access
to Y, while the rest are only weakly informed. We also sweep the informativeness of
the "informed" minority to study the "slightly more information than others" regime.
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
    ground_truth,
)

N_RUNS = 4000
PRIOR = 0.5
N_AGENTS = 12
WEAK_Q = 0.52  # the uninformed majority barely beats a coin flip


def experiment_a(rng) -> list[list]:
    """Vary the number of near-oracle agents among otherwise weak agents."""
    rows = []
    fracs, accs, gaps = [], [], []
    for n_oracle in range(0, N_AGENTS + 1):
        agents = []
        for _ in range(N_AGENTS - n_oracle):
            agents.append(Agent(binary_symmetric(WEAK_Q), TruthfulBayesian()))
        for _ in range(n_oracle):
            agents.append(Agent(ground_truth(0.99), TruthfulBayesian()))
        rng.shuffle(agents)  # randomise arrival order
        market = SelfResolvingMarket(MarketConfig(prior_p1=PRIOR, k=1))
        st = aggregation_stats(market.run_many(agents, N_RUNS, rng))
        frac = n_oracle / N_AGENTS
        fracs.append(frac)
        accs.append(st.accuracy)
        gaps.append(st.mean_abs_gap_to_full_info)
        rows.append([frac, n_oracle, st.accuracy, st.mean_brier, st.mean_total_cost])
    return rows, fracs, accs


def experiment_b(rng) -> tuple[list[list], list, list]:
    """One slightly-better-informed agent: sweep its informational edge."""
    rows, qualities, accs = [], [], []
    for q_informed in np.linspace(0.52, 0.95, 10):
        agents = [Agent(binary_symmetric(WEAK_Q), TruthfulBayesian())
                  for _ in range(N_AGENTS - 1)]
        agents.append(Agent(binary_symmetric(q_informed), TruthfulBayesian()))
        rng.shuffle(agents)
        market = SelfResolvingMarket(MarketConfig(prior_p1=PRIOR, k=1))
        st = aggregation_stats(market.run_many(agents, N_RUNS, rng))
        rows.append([round(float(q_informed), 3), st.accuracy, st.mean_brier])
        qualities.append(float(q_informed))
        accs.append(st.accuracy)
    return rows, qualities, accs


def main() -> None:
    rng = np.random.default_rng(1)
    rows_a, fracs, accs_a = experiment_a(rng)
    rows_b, quals, accs_b = experiment_b(rng)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot([f * 100 for f in fracs], accs_a, marker="o", color="tab:green")
    ax1.set(xlabel="% of agents with near-oracle access",
            ylabel="market accuracy",
            title=f"Accuracy vs ground-truth access\n({N_AGENTS} agents, weak q={WEAK_Q})")
    ax1.grid(alpha=0.3)

    ax2.plot(quals, accs_b, marker="s", color="tab:purple")
    ax2.axvline(WEAK_Q, ls="--", color="gray", label="crowd quality")
    ax2.set(xlabel="signal quality of the one informed agent",
            ylabel="market accuracy",
            title="Value of a single better-informed agent")
    ax2.legend()
    ax2.grid(alpha=0.3)

    save_fig(fig, "exp_ground_truth.png")
    write_csv("exp_ground_truth_oracle_count.csv",
              ["oracle_fraction", "n_oracle", "accuracy", "brier", "mean_total_cost"],
              rows_a)
    write_csv("exp_ground_truth_single_informed.csv",
              ["informed_quality", "accuracy", "brier"], rows_b)


if __name__ == "__main__":
    main()
