"""Agent reporting strategies.

A strategy maps an :class:`AgentContext` (prior, previous market report, the
agent's own signal and signal structure, the full history, an RNG) to a reported
belief ``[P(Y=0), P(Y=1)]``.

The benchmark is :class:`TruthfulBayesian`, which is the PBE of the mechanism.
The other strategies are deviations you can mix into a population to study how the
market's accuracy and the deviators' payoffs respond.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .beliefs import belief_from_logodds, belief_from_p1, logit
from .signals import SignalStructure


@dataclass
class AgentContext:
    prior_p1: float
    prev_report: np.ndarray  # [P(Y=0), P(Y=1)] reported by the previous agent
    own_signal: int
    signal_structure: SignalStructure
    history: list = field(default_factory=list)  # list of past reports (arrays)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    private_info: dict = field(default_factory=dict)  # extra info handed to agent

    @property
    def prev_logodds(self) -> float:
        return logit(self.prev_report[1])

    @property
    def own_llr(self) -> float:
        return self.signal_structure.log_likelihood_ratio(self.own_signal)

    @property
    def truthful_logodds(self) -> float:
        """Bayesian update: invert the previous report and add own signal."""
        return self.prev_logodds + self.own_llr


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def report(self, ctx: AgentContext) -> np.ndarray:
        ...


class TruthfulBayesian(Strategy):
    """Report the true Bayesian posterior. This is the equilibrium strategy."""

    name = "truthful"

    def report(self, ctx: AgentContext) -> np.ndarray:
        return belief_from_logodds(ctx.truthful_logodds)


class ScaledSignal(Strategy):
    """Report ``prev_logodds + gamma * own_llr``.

    ``gamma = 1`` is truthful. ``gamma > 1`` exaggerates the own signal
    (over-confidence — the profitable-looking deviation the paper analyses, since
    cross-entropy rewards certainty). ``0 < gamma < 1`` hedges toward the prior;
    ``gamma = 0`` ignores the signal entirely (herding on the previous report).
    """

    def __init__(self, gamma: float):
        self.gamma = float(gamma)
        self.name = f"scaled(gamma={gamma:g})"

    def report(self, ctx: AgentContext) -> np.ndarray:
        return belief_from_logodds(ctx.prev_logodds + self.gamma * ctx.own_llr)


class Herding(Strategy):
    """Ignore the private signal and copy the previous market report."""

    name = "herding"

    def report(self, ctx: AgentContext) -> np.ndarray:
        return ctx.prev_report.copy()


class ConstantReport(Strategy):
    """Always report a fixed ``P(Y=1) = c`` regardless of signal or history.

    With ``c`` equal to the prior this is the *uninformative* strategy whose
    equilibrium payout is zero under CE-MSR (Section 6, Appendix C).
    """

    def __init__(self, c: float):
        self.c = float(c)
        self.name = f"constant(c={c:g})"

    def report(self, ctx: AgentContext) -> np.ndarray:
        return belief_from_p1(self.c)


class PriorReport(Strategy):
    """Always report the common prior (a specific uninformative strategy)."""

    name = "report_prior"

    def report(self, ctx: AgentContext) -> np.ndarray:
        return belief_from_p1(ctx.prior_p1)


class NoisyTruthful(Strategy):
    """Truthful posterior plus Gaussian noise in log-odds space.

    Models a boundedly-rational or noisy reporter (a "noise trader").
    """

    def __init__(self, sigma: float):
        self.sigma = float(sigma)
        self.name = f"noisy(sigma={sigma:g})"

    def report(self, ctx: AgentContext) -> np.ndarray:
        noise = ctx.rng.normal(0.0, self.sigma)
        return belief_from_logodds(ctx.truthful_logodds + noise)


class Contrarian(Strategy):
    """Adversarial: report log-odds with the sign of the own signal flipped."""

    name = "contrarian"

    def report(self, ctx: AgentContext) -> np.ndarray:
        return belief_from_logodds(ctx.prev_logodds - ctx.own_llr)


class Overconfident(Strategy):
    """Truthful direction, but push the report toward the nearest extreme.

    ``strength`` in [0, 1]: 0 = truthful, 1 = report 0/1.
    """

    def __init__(self, strength: float):
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.name = f"overconfident(s={strength:g})"

    def report(self, ctx: AgentContext) -> np.ndarray:
        p1 = belief_from_logodds(ctx.truthful_logodds)[1]
        target = 1.0 if p1 >= 0.5 else 0.0
        return belief_from_p1(p1 + self.strength * (target - p1))
