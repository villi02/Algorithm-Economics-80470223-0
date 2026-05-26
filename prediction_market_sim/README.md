# Self-Resolving Prediction Market Simulator

An agent-based simulator for the mechanism in **"Self-Resolving Prediction Markets
for Unverifiable Outcomes"** (Srinivasan, Karger & Chen, 2025). It lets you run the
sequential, self-resolving market with configurable agents, signals, scoring rules
and strategies, and collect empirical results for extending the paper.

## What it models

A binary outcome `Y ∈ {0,1}` with a common prior. `M` agents arrive **in sequence**.
Each has a private signal that is **conditionally independent of the others given `Y`**
(the "informational substitutes" assumption). Each agent sees the previous market
report plus its own signal, forms a belief, and reports it. The market **terminates
with probability `α`** after each report; the **terminal agent is the reference**.
Agents `1..T−k` are paid the **cross-entropy market scoring rule** against the
reference, and the last `k` agents receive a **flat fee `R`**.

Because signals are conditionally independent, posterior log-odds are additive, so a
truthful Bayesian agent inverts the previous report and adds its own signal's
log-likelihood ratio. Under truthful play the terminal report equals the ideal
full-information Bayesian posterior — which the simulator verifies.

## Install & run

```bash
cd prediction_market_sim
pip install -r requirements.txt

# minimal end-to-end example
PYTHONPATH=. python examples/quickstart.py

# the four bundled experiments (figures + CSVs land in results/)
python experiments/exp_aggregation.py
python experiments/exp_ground_truth.py
python experiments/exp_strategies.py
python experiments/exp_incentives.py
```

## The knobs you asked for

| Want to change… | How |
| --- | --- |
| **Number of agents** | length of the `agents` list you pass to `market.run` |
| **Score function** | `MarketConfig(scoring_rule=...)` — `CrossEntropyMSR` (default), `CrossEntropyScore`, `OutcomeLogScore`, `OutcomeLogMSR`, or your own `ScoringRule` subclass |
| **Who has ground-truth access** | give chosen agents `ground_truth(reliability)` signal structures; others a weaker `binary_symmetric(q)` |
| **"Slightly more information"** | tune each agent's signal quality `q` (or `delta`) per-agent — see `exp_ground_truth.py` |
| **Different strategies** | set each `Agent`'s `strategy`: `TruthfulBayesian`, `ScaledSignal(gamma)`, `Herding`, `ConstantReport`, `PriorReport`, `NoisyTruthful`, `Contrarian`, `Overconfident`, or your own `Strategy` |
| **Give specific info to specific agents** | per-agent `signal_structure`, and the free-form `Agent(..., private_info={...})` dict that your custom strategy can read |
| **Random vs fixed market length** | `MarketConfig(alpha=...)` for random stopping (paper's design), `alpha=0` for a fixed number of agents |
| **Flat-fee tail** | `MarketConfig(k=..., flat_fee=R)` |

## Building a custom run

```python
import numpy as np
from srpm import (Agent, SelfResolvingMarket, MarketConfig, CrossEntropyMSR,
                  binary_symmetric, ground_truth, TruthfulBayesian, ScaledSignal,
                  aggregation_stats)

agents = (
    [Agent(binary_symmetric(0.6), TruthfulBayesian()) for _ in range(8)]
    + [Agent(binary_symmetric(0.6), ScaledSignal(2.0), name="over-reporter")]
    + [Agent(ground_truth(0.99), TruthfulBayesian(), name="oracle")]
)
market = SelfResolvingMarket(MarketConfig(prior_p1=0.5, k=1, flat_fee=0.0,
                                          scoring_rule=CrossEntropyMSR(), alpha=0.0))
rng = np.random.default_rng(0)
stats = aggregation_stats(market.run_many(agents, 5000, rng))
print(stats)
```

### Writing a new strategy

```python
from srpm import Strategy, belief_from_logodds

class HalfHedge(Strategy):
    name = "half_hedge"
    def report(self, ctx):
        # ctx exposes: prior_p1, prev_report, own_signal, signal_structure,
        # prev_logodds, own_llr, truthful_logodds, history, rng, private_info
        return belief_from_logodds(ctx.prev_logodds + 0.5 * ctx.own_llr)
```

### Writing a new scoring rule

```python
from srpm import ScoringRule

class MyRule(ScoringRule):
    name = "my_rule"
    def score(self, report, prev_report, reference, outcome=None):
        ...  # return a float payoff
```

## Package layout

```
srpm/
  signals.py      SignalStructure + constructors (binary_symmetric, ground_truth, ...)
  beliefs.py      log-odds / Bayesian aggregation utilities
  scoring.py      ScoringRule classes (CE-MSR and benchmarks)
  strategies.py   Strategy classes + AgentContext
  agents.py       Agent (signal structure + strategy + private_info)
  market.py       SelfResolvingMarket, MarketConfig, MarketResult
  metrics.py      accuracy / Brier / log-loss / payoff aggregation
  incentives.py   peer-prediction deviation analysis (Theorems 1-3)
experiments/      four runnable studies, each writes figures + CSV to results/
examples/         quickstart.py
```

## The four bundled experiments

1. **`exp_aggregation.py`** — market accuracy and gap to the ideal Bayesian posterior
   vs number of agents and signal quality. Confirms exact aggregation under truthful play.
2. **`exp_ground_truth.py`** — accuracy vs the *fraction* of near-oracle agents, and the
   value of a single agent with a small informational edge over the crowd.
3. **`exp_strategies.py`** — mixed populations (herders, noisy reporters, uninformative
   agents, over-reporters): effect on the aggregate and per-strategy payoffs.
4. **`exp_incentives.py`** — a focal agent's best-response deviation `gamma` as a function
   of the reference's number of informational substitutes `k`; reproduces the
   truthfulness result (`gamma* → 1` as `k` grows).
