"""The self-resolving prediction market mechanism.

A single run:

1. Draw the outcome ``Y`` from the prior.
2. Draw each agent's private signal from its signal structure.
3. Agents report **in order**; each sees the previous market report and its own
   signal. The market prior seeds the first "previous report".
4. The market terminates either after a fixed number of agents, or randomly with
   probability ``alpha`` after each report (the paper's preferred design).
5. The **terminal agent ``T``** is the reference. Agents ``1..T-k`` are paid the
   scoring rule against the terminal report; the last ``k`` agents get a flat fee
   ``R`` (they have no valid reference far enough ahead of them).

See :class:`MarketConfig` for the configurable knobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agents import Agent
from .beliefs import belief_from_p1, full_information_posterior
from .scoring import CrossEntropyMSR, ScoringRule
from .strategies import AgentContext


@dataclass
class MarketConfig:
    prior_p1: float = 0.5
    scoring_rule: ScoringRule = field(default_factory=CrossEntropyMSR)
    k: int = 1  # number of trailing agents paid a flat fee
    flat_fee: float = 0.0  # R, paid to the last k agents
    alpha: float = 0.0  # per-step termination prob; 0 => fixed length (all agents)
    min_agents: int = 2  # floor on participants when using random stopping


@dataclass
class MarketResult:
    outcome: int
    signals: list  # realised signal value per participating agent
    reports: np.ndarray  # shape (T, 2), each row [P(Y=0), P(Y=1)]
    payoffs: np.ndarray  # shape (T,), payment to each participating agent
    reference: np.ndarray  # terminal agent's report (the reference distribution)
    n_participants: int
    aggregate_p1: float  # terminal report's P(Y=1): the market's final prediction
    full_info_p1: float  # ideal Bayesian posterior given all participating signals
    agents: list = field(default_factory=list)  # the participating Agent objects

    @property
    def total_cost(self) -> float:
        return float(np.sum(self.payoffs))


class SelfResolvingMarket:
    def __init__(self, config: MarketConfig | None = None):
        self.config = config or MarketConfig()

    def _num_participants(self, n_agents: int, rng: np.random.Generator) -> int:
        cfg = self.config
        if cfg.alpha <= 0.0:
            return n_agents
        t = 0
        while t < n_agents:
            t += 1
            if t >= cfg.min_agents and rng.random() < cfg.alpha:
                break
        return t

    def run(self, agents: list[Agent], rng: np.random.Generator) -> MarketResult:
        cfg = self.config
        T = self._num_participants(len(agents), rng)
        participants = agents[:T]

        # 1-2. Draw outcome and signals.
        outcome = int(rng.random() < cfg.prior_p1)
        signals = [a.signal_structure.sample(outcome, rng) for a in participants]

        # 3. Sequential reports.
        reports = np.zeros((T, 2))
        prev_report = belief_from_p1(cfg.prior_p1)
        history: list[np.ndarray] = []
        for i, agent in enumerate(participants):
            ctx = AgentContext(
                prior_p1=cfg.prior_p1,
                prev_report=prev_report,
                own_signal=signals[i],
                signal_structure=agent.signal_structure,
                history=history,
                rng=rng,
                private_info=agent.private_info,
            )
            q = np.asarray(agent.strategy.report(ctx), dtype=float)
            q = q / q.sum()  # normalise defensively
            reports[i] = q
            history.append(q)
            prev_report = q

        reference = reports[T - 1]

        # 4-5. Payments.
        payoffs = np.zeros(T)
        prior_belief = belief_from_p1(cfg.prior_p1)
        for i in range(T):
            if i >= T - cfg.k:
                payoffs[i] = cfg.flat_fee
            else:
                prev = prior_belief if i == 0 else reports[i - 1]
                payoffs[i] = cfg.scoring_rule.score(
                    report=reports[i],
                    prev_report=prev,
                    reference=reference,
                    outcome=outcome,
                )

        full_info = full_information_posterior(
            cfg.prior_p1,
            [a.signal_structure for a in participants],
            signals,
        )

        return MarketResult(
            outcome=outcome,
            signals=signals,
            reports=reports,
            payoffs=payoffs,
            reference=reference,
            n_participants=T,
            aggregate_p1=float(reference[1]),
            full_info_p1=float(full_info[1]),
            agents=participants,
        )

    def run_many(
        self, agents: list[Agent], n_runs: int, rng: np.random.Generator
    ) -> list[MarketResult]:
        return [self.run(agents, rng) for _ in range(n_runs)]
