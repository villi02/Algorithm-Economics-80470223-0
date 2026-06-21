"""Counterfactual payout analysis for a single resolved market.

We treat the Yes-price series as a stream of sequential reports.  Each price
``p_t`` is a report ``q^(t) = [1 - p_t, p_t]`` over ``Y in {0, 1}``.  The
"reporter" at step ``t`` is whoever moved the price from ``q^(t-1)`` to ``q^(t)``;
their payout is the *market scoring rule* increment.

Two payment schemes are compared on the **same** reporters and transitions:

* **Self-resolving** (the paper):  CE-MSR against the reference ``r``,
  ``S_CEM = sum_i r_i * (log q^t_i - log q^{t-1}_i)``.  Needs no outcome.
* **Verifiable** (the benchmark): log-MSR against the realised outcome ``Y``,
  ``log q^t_Y - log q^{t-1}_Y``.  Needs ground truth.

Both telescope, so the designer's total cost over reporters ``1..R-1`` is just the
score of the last reporter minus the first — reported as ``total_self`` and
``total_verifiable``.  We also keep the per-transition payouts to compare *which*
reporters each scheme rewards most.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# Reuse the paper's scoring rules from the simulator package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from srpm.scoring import CrossEntropyMSR, OutcomeLogMSR  # noqa: E402

from .references import Reference  # noqa: E402

_EPS = 1e-6
_CEM = CrossEntropyMSR()
_VER = OutcomeLogMSR()


def build_events(
    times: list[int] | np.ndarray,
    prices: list[float] | np.ndarray,
    min_price_change: float = 0.0,
    min_gap_sec: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a raw price series to discrete *reporting events*.

    Keeps the first point, then a later point only if the price moved by at least
    ``min_price_change`` (default: any change, i.e. drop flat repeats) and at
    least ``min_gap_sec`` has elapsed.  Prices are clipped away from {0, 1}.
    """
    t = np.asarray(times, dtype=float)
    p = np.clip(np.asarray(prices, dtype=float), _EPS, 1 - _EPS)
    if len(t) == 0:
        return t, p
    order = np.argsort(t)
    t, p = t[order], p[order]

    keep_t, keep_p = [t[0]], [p[0]]
    for ti, pi in zip(t[1:], p[1:]):
        if abs(pi - keep_p[-1]) >= max(min_price_change, 1e-12) and (ti - keep_t[-1]) >= min_gap_sec:
            keep_t.append(ti)
            keep_p.append(pi)
    return np.array(keep_t), np.array(keep_p)


def k_pooled_payouts(analyses) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool per-transition payouts across markets, each tagged by its *effective k*.

    For a market with ``T`` reporting transitions, the transition at zero-based
    index ``i`` has effective ``k = T - i`` — the number of subsequent reporting
    events between it and the reference (the reference being one step past the
    final reporter).  So ``k = 1`` is the reporter immediately before the
    reference; ``k = T`` is the first reporter.  This counts *events*, which is
    only an **upper bound** on the paper's true ``k`` (the number of
    conditionally-independent informational substitutes), because adjacent price
    changes may come from the same trader or be uninformed noise.
    """
    ks, sp, vp = [], [], []
    for a in analyses:
        n = len(a.self_payouts)
        if n == 0:
            continue
        idx = np.arange(n)
        ks.append(n - idx)
        sp.append(a.self_payouts)
        vp.append(a.verifiable_payouts)
    if not ks:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(ks), np.concatenate(sp), np.concatenate(vp)


# Log-spaced effective-k bucket edges; bucket b covers [edge[b], edge[b+1]).
K_BUCKET_EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 10**6])
K_BUCKET_LABELS = ["1", "2-3", "4-7", "8-15", "16-31", "32-63",
                   "64-127", "128-255", "256-511", "512+"]


def k_bucket_metrics(analyses, min_obs: int = 20) -> list[dict]:
    """For one reference, compute per-bucket agreement metrics.

    Returns one dict per non-empty bucket: ``{k_label, n, pearson, sign_agree,
    mean_abs_diff}``.  Buckets with fewer than ``min_obs`` transitions get
    ``nan`` for correlation/sign-agree.
    """
    k, sp, vp = k_pooled_payouts(analyses)
    if len(k) == 0:
        return []
    bucket = np.clip(np.digitize(k, K_BUCKET_EDGES) - 1, 0, len(K_BUCKET_LABELS) - 1)
    rows = []
    for b, label in enumerate(K_BUCKET_LABELS):
        mask = bucket == b
        n = int(mask.sum())
        if n == 0:
            continue
        if n >= min_obs and np.std(sp[mask]) > 1e-12 and np.std(vp[mask]) > 1e-12:
            pearson = float(np.corrcoef(sp[mask], vp[mask])[0, 1])
            sign_agree = float(np.mean(np.sign(sp[mask]) == np.sign(vp[mask])))
        else:
            pearson = float("nan")
            sign_agree = float("nan")
        rows.append({
            "k_label": label,
            "n": n,
            "pearson": pearson,
            "sign_agree": sign_agree,
            "mean_abs_diff": float(np.mean(np.abs(sp[mask] - vp[mask]))),
        })
    return rows


def series_interestingness(times, prices, min_price_change: float = 0.0) -> float:
    """Score how *illustrative* a market's price path is for an example figure.

    Rewards markets whose price genuinely travels through the middle of [0, 1]
    (large range, time spent away from the 0/1 rails) and that have many reporting
    events; penalises degenerate near-0/1 markets that plot as a flat line.
    Returns 0 for series too short to plot.
    """
    t, p = build_events(times, prices, min_price_change)
    if len(p) < 10:
        return 0.0
    price_range = float(p.max() - p.min())
    mid_fraction = float(np.mean((p > 0.15) & (p < 0.85)))  # time off the rails
    return price_range * (0.2 + mid_fraction) * np.log1p(len(p))


@dataclass
class MarketAnalysis:
    market_id: str
    question: str
    reference_name: str
    outcome: int
    ref_price: float
    n_reporters: int
    # Per-transition arrays (length n_reporters - 1), aligned by index.
    self_payouts: np.ndarray = field(repr=False)
    verifiable_payouts: np.ndarray = field(repr=False)
    reporter_times: np.ndarray = field(repr=False)
    reporter_prices: np.ndarray = field(repr=False)
    # Scalar summaries.
    category: str = ""
    total_self: float = 0.0
    total_verifiable: float = 0.0
    payout_ratio: float = float("nan")        # total_self / total_verifiable
    pearson: float = float("nan")             # corr of per-transition payouts
    spearman: float = float("nan")            # rank corr ("who gets rewarded")
    top_decile_jaccard: float = float("nan")  # overlap of top-10% reporters
    argmax_match: bool = False                 # same single most-rewarded reporter?

    def to_summary_row(self) -> dict:
        return {
            "market_id": self.market_id,
            "question": self.question[:80],
            "category": self.category,
            "reference": self.reference_name,
            "outcome": self.outcome,
            "ref_price": round(self.ref_price, 4),
            "n_reporters": self.n_reporters,
            "n_transitions": len(self.self_payouts),
            "total_self": self.total_self,
            "total_verifiable": self.total_verifiable,
            "payout_ratio": self.payout_ratio,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "top_decile_jaccard": self.top_decile_jaccard,
            "argmax_match": int(self.argmax_match),
        }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a))
    return r


def _top_decile_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    k = max(1, n // 10)
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else float("nan")


def analyze_market(
    market_id: str,
    question: str,
    outcome: int,
    times: np.ndarray,
    prices: np.ndarray,
    reference: Reference,
    min_price_change: float = 0.0,
    min_gap_sec: float = 0.0,
    min_reporters: int = 5,
    category: str = "",
) -> MarketAnalysis | None:
    """Run the counterfactual for one market under one reference.

    Returns ``None`` if the (event, reference) split leaves too few reporters.
    """
    t, p = build_events(times, prices, min_price_change, min_gap_sec)
    if len(t) < min_reporters + 1:
        return None

    n_rep, ref_price = reference.resolve(t, p)
    n_rep = int(min(n_rep, len(t)))
    if n_rep < min_reporters:
        return None

    rt, rp = t[:n_rep], p[:n_rep]
    ref_price = float(np.clip(ref_price, _EPS, 1 - _EPS))
    r_vec = np.array([1 - ref_price, ref_price])

    # Reports q^(t) = [1-p, p]; transitions t = 1 .. n_rep-1.
    q = np.column_stack([1 - rp, rp])
    self_pay = np.array([
        _CEM.score(q[i], q[i - 1], r_vec) for i in range(1, n_rep)
    ])
    ver_pay = np.array([
        _VER.score(q[i], q[i - 1], r_vec, outcome) for i in range(1, n_rep)
    ])

    a = MarketAnalysis(
        market_id=market_id,
        question=question,
        reference_name=reference.name,
        outcome=outcome,
        ref_price=ref_price,
        n_reporters=n_rep,
        category=category,
        self_payouts=self_pay,
        verifiable_payouts=ver_pay,
        reporter_times=rt,
        reporter_prices=rp,
        total_self=float(np.sum(self_pay)),
        total_verifiable=float(np.sum(ver_pay)),
    )
    a.payout_ratio = (a.total_self / a.total_verifiable
                      if abs(a.total_verifiable) > 1e-12 else float("nan"))
    a.pearson = _safe_corr(self_pay, ver_pay)
    a.spearman = _safe_corr(_rank(self_pay), _rank(ver_pay))
    a.top_decile_jaccard = _top_decile_jaccard(self_pay, ver_pay)
    a.argmax_match = bool(np.argmax(self_pay) == np.argmax(ver_pay))
    return a
