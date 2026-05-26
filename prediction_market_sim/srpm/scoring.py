"""Scoring rules used to pay agents.

The paper's mechanism pays an agent with a *cross-entropy market scoring rule*
(CE-MSR) evaluated against a reference agent's prediction ``r`` rather than the
(unobservable) outcome:

    S_CEM(r, q_t, q_{t-1}) = -H(r, q_t) + H(r, q_{t-1})
                           = sum_i r_i * log( q_t,i / q_{t-1},i )

i.e. how much the agent's report ``q_t`` improved the cross-entropy against the
reference relative to the previous market report ``q_{t-1}``.

All scoring rules implement ``score(report, prev_report, reference, outcome)``.
``reference`` is the reference agent's distribution; ``outcome`` is the realised
``Y`` (only used by the outcome-based benchmark rules, which require ground truth
and so are *not* self-resolving — they exist purely for comparison).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_EPS = 1e-12


def _safe(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0)


class ScoringRule(ABC):
    name: str = "scoring_rule"

    @abstractmethod
    def score(
        self,
        report: np.ndarray,
        prev_report: np.ndarray,
        reference: np.ndarray,
        outcome: int | None = None,
    ) -> float:
        ...


class CrossEntropyMSR(ScoringRule):
    """The paper's preferred rule: CE market scoring against a reference agent."""

    name = "cross_entropy_msr"

    def score(self, report, prev_report, reference, outcome=None) -> float:
        r = _safe(reference)
        q_t = _safe(report)
        q_prev = _safe(prev_report)
        return float(np.sum(r * (np.log(q_t) - np.log(q_prev))))


class CrossEntropyScore(ScoringRule):
    """Plain (non-market) cross-entropy against the reference: ``-H(r, q_t)``.

    This is the "peer prediction" payment discussed in Section 5 — paying an agent
    purely on agreement with the reference, with no improvement term. It is *not*
    truthful in general (it rewards over-confident reports).
    """

    name = "cross_entropy"

    def score(self, report, prev_report, reference, outcome=None) -> float:
        r = _safe(reference)
        q_t = _safe(report)
        return float(np.sum(r * np.log(q_t)))


class OutcomeLogScore(ScoringRule):
    """Benchmark: log score against the realised outcome (requires ground truth).

    Useful for measuring calibration/accuracy, but NOT self-resolving.
    """

    name = "outcome_log_score"

    def score(self, report, prev_report, reference, outcome=None) -> float:
        if outcome is None:
            raise ValueError("OutcomeLogScore requires the realised outcome")
        q_t = _safe(report)
        return float(np.log(q_t[outcome]))


class OutcomeLogMSR(ScoringRule):
    """Benchmark: log market scoring rule against the realised outcome."""

    name = "outcome_log_msr"

    def score(self, report, prev_report, reference, outcome=None) -> float:
        if outcome is None:
            raise ValueError("OutcomeLogMSR requires the realised outcome")
        q_t = _safe(report)
        q_prev = _safe(prev_report)
        return float(np.log(q_t[outcome]) - np.log(q_prev[outcome]))
