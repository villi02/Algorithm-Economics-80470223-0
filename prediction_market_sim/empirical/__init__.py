"""Empirical counterfactual study on real Polymarket data.

This subpackage implements *Option 1* of the empirical plan for the paper
"Self-Resolving Prediction Markets for Unverifiable Outcomes": treat a resolved
market's price time series as a stream of sequential *reports* ``q^(t)``, pick a
late price (or a window of late prices) as the *reference* ``r``, and compute the
counterfactual cross-entropy market-scoring-rule (CE-MSR) payouts the
self-resolving mechanism *would* have paid each "reporter" using ``r`` instead of
the realised outcome.  We then compare those self-resolving payouts to the
verifiable (outcome-based) log-MSR payouts.

This is **not** a test of incentive compatibility (that needs a live mechanism).
It tests whether the self-resolving mechanism's *outputs* track the verifiable
mechanism's outputs on observational data.

Modules
-------
``polymarket``    Gamma + CLOB API client with on-disk caching.
``references``    Strategies for choosing the reference ``r`` (lead / fraction / window).
``counterfactual``Core per-market payout computation and summary metrics.
``plots``         Matplotlib figures.
``run_empirical`` CLI driver: fetch markets, run the reference sweep, write CSVs + figures.
"""
from .polymarket import MarketMeta, PolymarketClient
from .references import (
    Reference,
    LeadReference,
    FractionReference,
    WindowReference,
    default_reference_sweep,
)
from .counterfactual import MarketAnalysis, analyze_market

__all__ = [
    "MarketMeta",
    "PolymarketClient",
    "Reference",
    "LeadReference",
    "FractionReference",
    "WindowReference",
    "default_reference_sweep",
    "MarketAnalysis",
    "analyze_market",
]
