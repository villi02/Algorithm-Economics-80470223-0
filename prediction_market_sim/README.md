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
empirical/        real-Polymarket counterfactual study (reuses srpm.scoring)
  polymarket.py     Gamma + CLOB API client with on-disk caching
  references.py     reference-r strategies (lead / fraction / window) + sweep
  categorize.py     keyword topic classifier (politics / sports / crypto / ...)
  counterfactual.py per-market payout computation + summary metrics
  plots.py          figures
  run_empirical.py  CLI driver -> empirical/results/
```

## Empirical study on real Polymarket data (`empirical/`)

The `empirical/` subpackage implements **Option 1** of the empirical plan: rather
than simulate agents, it treats a *resolved* Polymarket market's price time series
as a stream of sequential reports and asks **how closely the self-resolving
mechanism's payouts would have tracked the verifiable (outcome-based) mechanism's
payouts** on real data.

For each market:

- Each Yes-price `p_t` is a report `q^(t) = [1−p_t, p_t]`; a *reporting event* is a
  price change (consecutive flat repeats are dropped).
- A **reference `r`** is a *late* price (the empirical analogue of the paper's
  terminal agent). "Late" mixes two channels, so we sweep several references:
  - `lead_1h` — price ~1h before the last data point: an **upper bound** dominated
    by *outcome leakage* (the question is nearly settled).
  - `lead_24h` — a moderately late midpoint.
  - `lead_168h` (1 week) and `life_50pct` (halfway through the market's life) —
    well-informed but **not omniscient** forecasts, closest to the paper's spirit
    and the most honest basis for the empirical claim.
  - `window_1-24h` — the *average* of late prices, the variance-reducing
    multi-reference design from the paper's Section 6.2.
- Each reporter (price transition) is paid two ways on the **same** transitions:
  - **self-resolving**: CE-MSR against `r` (`srpm.scoring.CrossEntropyMSR`) — no
    outcome needed;
  - **verifiable**: log-MSR against the realised `Y` (`srpm.scoring.OutcomeLogMSR`).
- We compare the two: total payout to the designer, Pearson/Spearman correlation
  of per-reporter payouts, top-decile reporter overlap, and whether the single
  most-rewarded reporter matches.

```bash
cd prediction_market_sim
pip install -r requirements.txt          # now also needs `requests`

python empirical/run_empirical.py                       # ~200 top-volume markets
python empirical/run_empirical.py --n 100 --min-volume 50000
python empirical/run_empirical.py --category politics --refresh
python empirical/run_empirical.py --help                # all knobs
```

Outputs land in `empirical/results/`:

- `per_market.csv` — one row per market × reference (includes the inferred topic).
- `reference_sweep.csv` — medians per reference across all markets.
- `category_breakdown.csv` — medians per (topic, reference).
- `fig_reference_sweep.png` — **the headline plot**: how agreement & the payout
  ratio move as `r` gets earlier.
- `fig_category_sweep.png` — the same, split by topic.
- per-reference `fig_total_scatter_*`, `fig_pooled_*`, `fig_hist_*`.
- `fig_example_*` — timeline panels for the most *illustrative* markets (price
  genuinely travels through the middle, balanced across Yes/No outcomes).

**Topics** are inferred by a small reproducible keyword classifier
(`empirical/categorize.py`: politics / sports / crypto / economy / tech / …),
because Gamma's own `category`/`tags` fields are empty for most markets. API
responses are cached under `empirical/cache/` (git-ignored; delete to refresh).

**How to read it.** As the reference moves from near-resolution (`lead_1h`) to
well before it (`life_50pct`), agreement with the verifiable scheme falls — that
gap *is* the outcome-leakage channel. What remains is how well a genuinely
forecast-only reference reproduces ground-truth incentives. Frame the claim
honestly: this measures whether the mechanism's **outputs track** the verifiable
mechanism's, **not** incentive compatibility (which needs a live mechanism where
agents can deviate).

## The four bundled (synthetic) experiments

1. **`exp_aggregation.py`** — market accuracy and gap to the ideal Bayesian posterior
   vs number of agents and signal quality. Confirms exact aggregation under truthful play.
2. **`exp_ground_truth.py`** — accuracy vs the *fraction* of near-oracle agents, and the
   value of a single agent with a small informational edge over the crowd.
3. **`exp_strategies.py`** — mixed populations (herders, noisy reporters, uninformative
   agents, over-reporters): effect on the aggregate and per-strategy payoffs.
4. **`exp_incentives.py`** — a focal agent's best-response deviation `gamma` as a function
   of the reference's number of informational substitutes `k`; reproduces the
   truthfulness result (`gamma* → 1` as `k` grows).
