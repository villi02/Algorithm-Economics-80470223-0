"""Signal structures and signal generation.

The outcome ``Y`` is binary in ``{0, 1}``. Each agent observes a private signal
drawn from a *signal structure*: a conditional probability table ``P(X | Y)`` over
a finite signal space. Signals are assumed conditionally independent given ``Y``
(Assumption 4, "informational substitutes", in Srinivasan, Karger & Chen 2025).

The informativeness of a structure is summarised by its Bhattacharyya coefficient
``BC = sum_s sqrt(P(s|0) P(s|1))`` and ``delta = 1 - BC`` (Definition 1). A larger
``delta`` means a more informative signal; ``delta = 0`` is uninformative and
``delta -> 1`` approaches perfect access to the ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalStructure:
    """A conditional probability table ``P(X = s | Y = y)`` for one agent.

    Parameters
    ----------
    cpt:
        Array of shape ``(2, n_signals)`` where ``cpt[y, s] = P(X = s | Y = y)``.
        Each row must sum to 1.
    name:
        Human-readable label, used in plots and logs.
    """

    cpt: np.ndarray
    name: str = ""
    llr: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.cpt = np.asarray(self.cpt, dtype=float)
        if self.cpt.shape[0] != 2:
            raise ValueError("cpt must have shape (2, n_signals)")
        if not np.allclose(self.cpt.sum(axis=1), 1.0):
            raise ValueError("each row of cpt must sum to 1")
        # Log-likelihood ratio of each signal value: log P(s|Y=1) / P(s|Y=0).
        with np.errstate(divide="ignore"):
            self.llr = np.log(self.cpt[1]) - np.log(self.cpt[0])

    @property
    def n_signals(self) -> int:
        return self.cpt.shape[1]

    def sample(self, y: int, rng: np.random.Generator) -> int:
        """Draw a signal value given the realised outcome ``y``."""
        return int(rng.choice(self.n_signals, p=self.cpt[y]))

    def sample_many(
        self, outcomes: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Vectorised draw: one signal per entry of ``outcomes`` (array of 0/1)."""
        outcomes = np.asarray(outcomes, dtype=int)
        cdf = np.cumsum(self.cpt, axis=1)  # (2, n_signals)
        u = rng.random(outcomes.shape[0])
        # For each sample, find first signal index where u < cdf[outcome].
        return (u[:, None] >= cdf[outcomes]).sum(axis=1)

    def log_likelihood_ratio(self, signal: int) -> float:
        """Evidence (in log-odds) that this signal value contributes about Y=1."""
        return float(self.llr[signal])

    @property
    def bhattacharyya_coefficient(self) -> float:
        return float(np.sum(np.sqrt(self.cpt[0] * self.cpt[1])))

    @property
    def delta(self) -> float:
        """Informativeness ``1 - BC``; 0 = uninformative, ->1 = near ground truth."""
        return 1.0 - self.bhattacharyya_coefficient


def binary_symmetric(quality: float, name: str = "") -> SignalStructure:
    """Binary signal that matches the outcome with probability ``quality``.

    ``P(X=1|Y=1) = P(X=0|Y=0) = quality``. ``quality = 0.5`` is uninformative,
    ``quality -> 1`` approaches perfect knowledge of ``Y``.
    """
    q = float(np.clip(quality, 1e-6, 1 - 1e-6))
    cpt = np.array([[q, 1 - q], [1 - q, q]])
    return SignalStructure(cpt=cpt, name=name or f"binary(q={quality:.3f})")


def binary_asymmetric(
    p1_given_1: float, p0_given_0: float, name: str = ""
) -> SignalStructure:
    """Binary signal with possibly different accuracy on the two outcomes."""
    a = float(np.clip(p1_given_1, 1e-6, 1 - 1e-6))
    b = float(np.clip(p0_given_0, 1e-6, 1 - 1e-6))
    # Row y holds [P(X=0|y), P(X=1|y)].
    cpt = np.array([[b, 1 - b], [1 - a, a]])
    return SignalStructure(cpt=cpt, name=name or f"binary({a:.2f},{b:.2f})")


def ground_truth(reliability: float = 0.999, name: str = "ground_truth") -> SignalStructure:
    """A near-perfect signal: an agent who (almost) observes ``Y`` directly.

    ``reliability`` is capped just below 1 so log-likelihood ratios stay finite.
    """
    return binary_symmetric(reliability, name=name)


def uninformative(name: str = "uninformative") -> SignalStructure:
    """A signal carrying no information about ``Y`` (BC = 1, delta = 0)."""
    return binary_symmetric(0.5, name=name)
