"""Bayesian belief utilities for a binary outcome.

Beliefs are represented as 2-vectors ``[P(Y=0), P(Y=1)]``. Because signals are
conditionally independent given ``Y``, posterior **log-odds** are additive:

    logit P(Y=1 | x_1..x_t) = logit(prior) + sum_j LLR_j(x_j)

A truthful Bayesian agent who observes the previous report ``q_{t-1}`` can invert
it (stochastic relevance, Assumption 3) to recover the running log-odds of all
earlier signals, then simply add the log-likelihood ratio of its own signal.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def logit(p1: float) -> float:
    """Log-odds of ``P(Y=1) = p1``."""
    p1 = float(np.clip(p1, _EPS, 1 - _EPS))
    return float(np.log(p1) - np.log(1 - p1))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def belief_from_p1(p1: float) -> np.ndarray:
    """Build the 2-vector ``[P(Y=0), P(Y=1)]`` from ``P(Y=1)``."""
    p1 = float(np.clip(p1, _EPS, 1 - _EPS))
    return np.array([1 - p1, p1])


def belief_from_logodds(x: float) -> np.ndarray:
    return belief_from_p1(sigmoid(x))


def full_information_posterior(
    prior_p1: float, signal_structures, signals
) -> np.ndarray:
    """The ideal aggregate: Bayesian posterior given the prior and *all* signals.

    This is what a perfectly truthful market should converge to (the benchmark
    that the terminal agent's report is compared against).
    """
    x = logit(prior_p1)
    for ss, s in zip(signal_structures, signals):
        x += ss.log_likelihood_ratio(s)
    return belief_from_logodds(x)
