"""Experiment 3: Mixed populations of strategies.

How do non-truthful agents affect the market's aggregate prediction and the payoffs
each strategy earns? We compare a fully-truthful baseline against populations laced
with herders, noisy reporters and uninformative (constant-prior) agents.

A central claim of the paper: uninformative reports earn ~0 under CE-MSR (they never
improve on the previous report), so honesty is also the most *profitable* behaviour.
"""
from __future__ import annotations

import numpy as np

from _common import save_fig, write_csv
import matplotlib.pyplot as plt

from srpm import (
    Agent,
    ConstantReport,
    Herding,
    MarketConfig,
    NoisyTruthful,
    PriorReport,
    ScaledSignal,
    SelfResolvingMarket,
    TruthfulBayesian,
    aggregation_stats,
    binary_symmetric,
)

N_RUNS = 6000
PRIOR = 0.5
N = 12
Q = 0.65


def make_population(kind: str) -> list[Agent]:
    ss = lambda: binary_symmetric(Q)
    if kind == "all_truthful":
        strats = [TruthfulBayesian() for _ in range(N)]
    elif kind == "half_herders":
        strats = [TruthfulBayesian() if i % 2 else Herding() for i in range(N)]
    elif kind == "noisy":
        strats = [NoisyTruthful(1.0) for _ in range(N)]
    elif kind == "uninformative_third":
        strats = [PriorReport() if i % 3 == 0 else TruthfulBayesian()
                  for i in range(N)]
    elif kind == "overconfident":
        strats = [ScaledSignal(2.5) for _ in range(N)]
    else:
        raise ValueError(kind)
    return [Agent(ss(), s) for s in strats]


def per_strategy_payoffs(results) -> dict[str, float]:
    """Average payoff grouped by the strategy name at each position."""
    sums: dict[str, list[float]] = {}
    for r in results:
        for i, ag in enumerate(r.agents):
            sums.setdefault(ag.strategy.name, []).append(r.payoffs[i])
    return {k: float(np.mean(v)) for k, v in sums.items()}


def main() -> None:
    rng = np.random.default_rng(2)
    kinds = ["all_truthful", "half_herders", "noisy",
             "uninformative_third", "overconfident"]

    rows, labels, briers, gaps = [], [], [], []
    payoff_report_lines = []
    for kind in kinds:
        agents = make_population(kind)
        market = SelfResolvingMarket(MarketConfig(prior_p1=PRIOR, k=1))
        results = market.run_many(agents, N_RUNS, rng)
        st = aggregation_stats(results)
        labels.append(kind)
        briers.append(st.mean_brier)
        gaps.append(st.mean_abs_gap_to_full_info)
        rows.append([kind, st.accuracy, st.mean_brier,
                     st.mean_abs_gap_to_full_info, st.mean_total_cost])
        payoffs = per_strategy_payoffs(results)
        payoff_report_lines.append(
            f"  {kind:20s}: " + ", ".join(f"{k}={v:+.4f}" for k, v in payoffs.items())
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(labels, briers, color="tab:blue", alpha=0.8)
    ax1.set(ylabel="Brier score (lower=better)",
            title="Aggregate accuracy by population")
    ax1.tick_params(axis="x", rotation=25)
    ax2.bar(labels, gaps, color="tab:red", alpha=0.8)
    ax2.set(ylabel="|market - full-info posterior|",
            title="Deviation from ideal aggregate")
    ax2.tick_params(axis="x", rotation=25)

    save_fig(fig, "exp_strategies.png")
    write_csv("exp_strategies.csv",
              ["population", "accuracy", "brier", "gap_to_full_info",
               "mean_total_cost"], rows)
    print("Mean payoff by strategy within each population:")
    print("\n".join(payoff_report_lines))


if __name__ == "__main__":
    main()
