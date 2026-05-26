"""Minimal end-to-end example: build a market, run it, inspect one outcome."""
import numpy as np

from srpm import (
    Agent,
    MarketConfig,
    SelfResolvingMarket,
    TruthfulBayesian,
    aggregation_stats,
    binary_symmetric,
    ground_truth,
    mean_payoff_for_agent,
)


def main() -> None:
    rng = np.random.default_rng(42)

    # A population: nine moderately-informed agents and one near-oracle.
    agents = [Agent(binary_symmetric(0.6), TruthfulBayesian()) for _ in range(9)]
    agents.append(Agent(ground_truth(0.99), TruthfulBayesian(), name="oracle"))

    market = SelfResolvingMarket(MarketConfig(prior_p1=0.5, k=1, flat_fee=0.0))

    # A single run, shown step by step.
    res = market.run(agents, rng)
    print(f"Realised outcome Y = {res.outcome}")
    print(f"Participants       = {res.n_participants}")
    print("Sequential reports P(Y=1):")
    for i, (rep, pay) in enumerate(zip(res.reports, res.payoffs)):
        print(f"  agent {i:2d} [{agents[i].name:>22}] -> {rep[1]:.3f}  payoff={pay:+.4f}")
    print(f"Market prediction  = {res.aggregate_p1:.3f}")
    print(f"Full-info posterior= {res.full_info_p1:.3f}")

    # Many runs: aggregate statistics.
    results = market.run_many(agents, 5000, rng)
    stats = aggregation_stats(results)
    print("\n--- 5000 runs ---")
    print(f"Brier (market)     = {stats.mean_brier:.4f}")
    print(f"Brier (full info)  = {stats.mean_brier_full_info:.4f}")
    print(f"Gap to full info   = {stats.mean_abs_gap_to_full_info:.4f}")
    print(f"Accuracy           = {stats.accuracy:.3f}")
    print(f"Mean total cost    = {stats.mean_total_cost:.4f}")
    print(f"Mean payoff agent 0= {mean_payoff_for_agent(results, 0):+.4f}")


if __name__ == "__main__":
    main()
