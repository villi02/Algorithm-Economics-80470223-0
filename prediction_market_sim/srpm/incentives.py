"""Incentive analysis: does a focal agent gain by deviating from truthful reporting?

This implements the Section 5 "peer prediction under cross-entropy" setup directly,
which is the cleanest way to reproduce the paper's truthfulness theorems.

A focal agent ``t`` with signal structure ``focal_ss`` observes a signal and reports
``q_t``. A reference agent ``r`` observes ``q_t`` (inverting it to the implied signal)
and *also* has ``k`` of its own private signals that are informational substitutes
(conditionally independent given ``Y``, not visible to the focal agent). The reference
forms a posterior ``r`` and the focal agent is paid the cross-entropy score
``S_CE(r, q_t) = sum_i r_i log q_t,i``.

Key prediction (Theorems 1-3): with few substitutes (small ``k``) the focal agent
profits by **over-reporting** (pushing its report toward certainty, ``gamma > 1``),
because cross-entropy rewards confident agreement. As ``k`` grows, the reference's
own signals dominate and its belief converges to the ground truth, so the deviator's
influence "washes out" and the expected payoff is maximised at **truthful** reporting
(``gamma = 1``).

The deviation family used here is :class:`~srpm.strategies.ScaledSignal`: the focal
agent reports log-odds ``logit(prior) + gamma * own_llr``.
"""
from __future__ import annotations

import numpy as np

from .beliefs import logit, sigmoid
from .scoring import CrossEntropyScore, ScoringRule
from .signals import SignalStructure


def _belief(p1: float) -> np.ndarray:
    p1 = float(np.clip(p1, 1e-12, 1 - 1e-12))
    return np.array([1 - p1, p1])


def expected_payoff_vs_gamma(
    prior_p1: float,
    focal_ss: SignalStructure,
    ref_structures: list[SignalStructure],
    gammas: np.ndarray,
    n_samples: int = 200_000,
    rng: np.random.Generator | None = None,
    scoring: ScoringRule | None = None,
) -> np.ndarray:
    """Monte-Carlo expected payoff of the focal agent for each deviation ``gamma``.

    Returns an array aligned with ``gammas``. ``gamma = 1`` is truthful reporting.
    ``len(ref_structures)`` is the number ``k`` of reference informational substitutes.
    """
    rng = rng or np.random.default_rng()
    scoring = scoring or CrossEntropyScore()
    prior_lo = logit(prior_p1)
    gammas = np.asarray(gammas, dtype=float)

    # Sample the world and signals once; reuse across all gammas for low variance.
    outcomes = (rng.random(n_samples) < prior_p1).astype(int)
    focal_signals = focal_ss.sample_many(outcomes, rng)
    focal_llr = focal_ss.llr[focal_signals]

    # Reference's own k independent signals -> summed log-likelihood ratio.
    ref_llr_sum = np.zeros(n_samples)
    for ss in ref_structures:
        sig = ss.sample_many(outcomes, rng)
        ref_llr_sum += ss.llr[sig]

    payoffs = np.zeros(len(gammas))
    prior_belief = _belief(prior_p1)
    for gi, gamma in enumerate(gammas):
        q_lo = prior_lo + gamma * focal_llr          # focal report log-odds
        q1 = 1.0 / (1.0 + np.exp(-q_lo))
        # Reference inverts the report (claimed signal) and adds its own signals.
        r_lo = q_lo + ref_llr_sum
        r1 = 1.0 / (1.0 + np.exp(-r_lo))
        # Cross-entropy score S_CE(r, q_t) = r1*log q1 + r0*log q0.
        q1c = np.clip(q1, 1e-12, 1 - 1e-12)
        if scoring.name == "cross_entropy":
            s = r1 * np.log(q1c) + (1 - r1) * np.log(1 - q1c)
        else:
            # Fall back to the generic per-sample scoring interface.
            s = np.array(
                [
                    scoring.score(
                        report=_belief(q1[i]),
                        prev_report=prior_belief,
                        reference=_belief(r1[i]),
                        outcome=int(outcomes[i]),
                    )
                    for i in range(n_samples)
                ]
            )
        payoffs[gi] = float(np.mean(s))
    return payoffs


def optimal_gamma(gammas: np.ndarray, payoffs: np.ndarray) -> float:
    """The deviation that maximises expected payoff (1.0 means truthful is best)."""
    return float(np.asarray(gammas)[int(np.argmax(payoffs))])
