"""Self-Resolving Prediction Market simulator.

A small, modular agent-based simulator for the mechanism in
"Self-Resolving Prediction Markets for Unverifiable Outcomes"
(Srinivasan, Karger & Chen, 2025).

Quick start
-----------
>>> import numpy as np
>>> from srpm import (Agent, SelfResolvingMarket, MarketConfig,
...                   binary_symmetric, TruthfulBayesian, aggregation_stats)
>>> agents = [Agent(binary_symmetric(0.6), TruthfulBayesian()) for _ in range(10)]
>>> market = SelfResolvingMarket(MarketConfig(prior_p1=0.5, k=1))
>>> rng = np.random.default_rng(0)
>>> results = market.run_many(agents, 2000, rng)
>>> stats = aggregation_stats(results)
"""
from .agents import Agent
from .beliefs import (
    belief_from_logodds,
    belief_from_p1,
    full_information_posterior,
    logit,
    sigmoid,
)
from .incentives import expected_payoff_vs_gamma, optimal_gamma
from .market import MarketConfig, MarketResult, SelfResolvingMarket
from .metrics import (
    AggregationStats,
    aggregation_stats,
    brier_score,
    log_loss,
    mean_payoff_for_agent,
    payoff_std_for_agent,
)
from .scoring import (
    CrossEntropyMSR,
    CrossEntropyScore,
    OutcomeLogMSR,
    OutcomeLogScore,
    ScoringRule,
)
from .signals import (
    SignalStructure,
    binary_asymmetric,
    binary_symmetric,
    ground_truth,
    uninformative,
)
from .strategies import (
    AgentContext,
    ConstantReport,
    Contrarian,
    Herding,
    NoisyTruthful,
    Overconfident,
    PriorReport,
    ScaledSignal,
    Strategy,
    TruthfulBayesian,
)

__all__ = [
    "Agent",
    "SelfResolvingMarket",
    "MarketConfig",
    "MarketResult",
    "SignalStructure",
    "binary_symmetric",
    "binary_asymmetric",
    "ground_truth",
    "uninformative",
    "ScoringRule",
    "CrossEntropyMSR",
    "CrossEntropyScore",
    "OutcomeLogScore",
    "OutcomeLogMSR",
    "Strategy",
    "AgentContext",
    "TruthfulBayesian",
    "ScaledSignal",
    "Herding",
    "ConstantReport",
    "PriorReport",
    "NoisyTruthful",
    "Contrarian",
    "Overconfident",
    "aggregation_stats",
    "AggregationStats",
    "brier_score",
    "log_loss",
    "mean_payoff_for_agent",
    "payoff_std_for_agent",
    "logit",
    "sigmoid",
    "belief_from_p1",
    "belief_from_logodds",
    "full_information_posterior",
    "expected_payoff_vs_gamma",
    "optimal_gamma",
]
