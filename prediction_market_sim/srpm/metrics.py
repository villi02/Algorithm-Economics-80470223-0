"""Aggregation, accuracy, and incentive metrics computed over many market runs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import MarketResult


def brier_score(p1: float, outcome: int) -> float:
    """Squared error of a probabilistic forecast (lower is better)."""
    return float((p1 - outcome) ** 2)


def log_loss(p1: float, outcome: int, eps: float = 1e-12) -> float:
    p1 = float(np.clip(p1, eps, 1 - eps))
    return float(-(outcome * np.log(p1) + (1 - outcome) * np.log(1 - p1)))


@dataclass
class AggregationStats:
    n_runs: int
    mean_brier: float          # market prediction vs realised outcome
    mean_log_loss: float
    mean_brier_full_info: float  # ideal full-info posterior vs outcome (benchmark)
    mean_abs_gap_to_full_info: float  # |market p1 - full-info p1|
    accuracy: float            # fraction where round(market p1) == outcome
    mean_total_cost: float     # mean total payout by the mechanism


def aggregation_stats(results: list[MarketResult]) -> AggregationStats:
    briers, losses, fi_briers, gaps, correct, costs = [], [], [], [], [], []
    for r in results:
        briers.append(brier_score(r.aggregate_p1, r.outcome))
        losses.append(log_loss(r.aggregate_p1, r.outcome))
        fi_briers.append(brier_score(r.full_info_p1, r.outcome))
        gaps.append(abs(r.aggregate_p1 - r.full_info_p1))
        correct.append(int(round(r.aggregate_p1) == r.outcome))
        costs.append(r.total_cost)
    return AggregationStats(
        n_runs=len(results),
        mean_brier=float(np.mean(briers)),
        mean_log_loss=float(np.mean(losses)),
        mean_brier_full_info=float(np.mean(fi_briers)),
        mean_abs_gap_to_full_info=float(np.mean(gaps)),
        accuracy=float(np.mean(correct)),
        mean_total_cost=float(np.mean(costs)),
    )


def mean_payoff_for_agent(results: list[MarketResult], index: int) -> float:
    """Average payoff received by the agent at a given position across runs.

    Runs in which that position did not participate (random stopping ended early)
    are skipped.
    """
    vals = [r.payoffs[index] for r in results if r.n_participants > index]
    return float(np.mean(vals)) if vals else float("nan")


def payoff_std_for_agent(results: list[MarketResult], index: int) -> float:
    vals = [r.payoffs[index] for r in results if r.n_participants > index]
    return float(np.std(vals)) if vals else float("nan")
