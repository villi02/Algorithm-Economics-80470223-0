"""Figures for the empirical counterfactual study.

All functions take already-computed :class:`MarketAnalysis` objects (or summary
rows) and write a PNG.  They are deliberately matplotlib-only and headless.
"""
from __future__ import annotations

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .counterfactual import MarketAnalysis, K_BUCKET_LABELS, k_bucket_metrics  # noqa: E402


def _dt(ts):
    return [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]


def plot_market_timeline(a: MarketAnalysis, ref_price: float, path: str):
    """Three stacked panels for one market+reference: price path, per-transition
    payouts under each scheme, and cumulative designer cost."""
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    tt = _dt(a.reporter_times)
    ax0.plot(tt, a.reporter_prices, lw=1.2, color="#1f77b4", label="Yes price q(t)")
    ax0.axhline(ref_price, ls="--", color="#d62728", lw=1.2,
                label=f"reference r = {ref_price:.3f}")
    ax0.axhline(a.outcome, ls=":", color="#2ca02c", lw=1.2,
                label=f"realised outcome Y = {a.outcome}")
    ax0.set_ylabel("P(Yes)")
    ax0.set_ylim(-0.03, 1.03)
    ax0.legend(loc="best", fontsize=8)
    ax0.set_title(f"{a.question[:90]}\nreference = {a.reference_name}", fontsize=10)

    tt1 = tt[1:]
    ax1.plot(tt1, a.self_payouts, lw=0.9, color="#9467bd",
             label="self-resolving (CE-MSR vs r)")
    ax1.plot(tt1, a.verifiable_payouts, lw=0.9, color="#ff7f0e", alpha=0.8,
             label="verifiable (log-MSR vs Y)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("per-reporter payout")
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(tt1, np.cumsum(a.self_payouts), lw=1.4, color="#9467bd",
             label=f"self total = {a.total_self:.3f}")
    ax2.plot(tt1, np.cumsum(a.verifiable_payouts), lw=1.4, color="#ff7f0e",
             label=f"verifiable total = {a.total_verifiable:.3f}")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("cumulative designer cost")
    ax2.set_xlabel("time (UTC)")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_total_scatter(analyses: list[MarketAnalysis], path: str, title: str):
    """One point per market: verifiable total vs self-resolving total payout."""
    x = np.array([a.total_verifiable for a in analyses])
    y = np.array([a.total_self for a in analyses])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(x, y, s=28, alpha=0.7, color="#9467bd", edgecolor="k", linewidth=0.3)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, ls="--", color="k", lw=1, label="y = x")
    if len(x) > 2 and np.std(x) > 1e-9:
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.92, f"Pearson r = {r:.3f}\nN = {len(x)} markets",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("verifiable total payout  (vs realised outcome)")
    ax.set_ylabel("self-resolving total payout  (vs reference r)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pooled_transitions(analyses: list[MarketAnalysis], path: str, title: str):
    """All per-reporter transitions pooled: self vs verifiable payout, by outcome."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for outcome, color in ((1, "#2ca02c"), (0, "#d62728")):
        xs, ys = [], []
        for a in analyses:
            if a.outcome == outcome:
                xs.append(a.verifiable_payouts)
                ys.append(a.self_payouts)
        if xs:
            ax.scatter(np.concatenate(xs), np.concatenate(ys), s=5, alpha=0.25,
                       color=color, label=f"Y = {outcome}")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("verifiable per-reporter payout")
    ax.set_ylabel("self-resolving per-reporter payout")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reference_sweep(sweep: dict[str, list[MarketAnalysis]], path: str):
    """How agreement metrics change as the reference moves earlier (the two
    channels: outcome leakage vs genuine forecast information).

    ``sweep`` maps reference name -> list of MarketAnalysis across markets.
    """
    names = list(sweep.keys())
    median_pearson, median_spearman, median_ratio, median_jacc = [], [], [], []
    for n in names:
        a = sweep[n]
        median_pearson.append(np.nanmedian([x.pearson for x in a]))
        median_spearman.append(np.nanmedian([x.spearman for x in a]))
        median_jacc.append(np.nanmedian([x.top_decile_jaccard for x in a]))
        ratios = np.array([x.payout_ratio for x in a], dtype=float)
        median_ratio.append(np.nanmedian(ratios))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    xpos = np.arange(len(names))
    ax0.plot(xpos, median_pearson, "o-", label="Pearson (payout level)")
    ax0.plot(xpos, median_spearman, "s-", label="Spearman (payout rank)")
    ax0.plot(xpos, median_jacc, "^-", label="top-decile Jaccard")
    ax0.set_xticks(xpos)
    ax0.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax0.set_ylabel("median across markets")
    ax0.set_title("Agreement: self-resolving vs verifiable payouts", fontsize=10)
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    ax1.plot(xpos, median_ratio, "o-", color="#9467bd")
    ax1.axhline(1.0, ls="--", color="k", lw=1)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("median total_self / total_verifiable")
    ax1.set_title("Designer total payout ratio", fontsize=10)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_k_bucket_agreement(
    sweep: dict[str, list[MarketAnalysis]],
    ref_names: list[str],
    path: str,
):
    """Per-reporter agreement as a function of *effective k* — the number of
    subsequent reporting events between a reporter and the reference.

    For each reference in the sweep, every transition across all markets is
    pooled, bucketed by its effective ``k``, and within each bucket we compute
    pooled Pearson (level agreement) and the fraction of transitions where both
    schemes assign the same sign of payout (qualitative agreement).  This is the
    empirical analogue of the paper's ``k`` axis: as ``k`` shrinks (right side
    of each plot), the focal reporter sits ever-closer to the reference and the
    two schemes have more room to disagree.

    *Effective k counts events, which upper-bounds the paper's k (some events may
    be noise / non-independent), so the absolute thresholds are not directly
    comparable to Theorems 2 and 4.*
    """
    xpos = np.arange(len(K_BUCKET_LABELS))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    for ref_name in ref_names:
        a = sweep.get(ref_name, [])
        if not a:
            continue
        rows = k_bucket_metrics(a)
        if not rows:
            continue
        by_label = {r["k_label"]: r for r in rows}
        pear = [by_label.get(lbl, {}).get("pearson", np.nan) for lbl in K_BUCKET_LABELS]
        sign = [by_label.get(lbl, {}).get("sign_agree", np.nan) for lbl in K_BUCKET_LABELS]
        ax0.plot(xpos, pear, "o-", label=ref_name)
        ax1.plot(xpos, sign, "o-", label=ref_name)

    for ax, ylabel, title in (
        (ax0, "pooled Pearson corr (self vs verifiable)", "Level agreement vs effective k"),
        (ax1, "fraction sign(self) == sign(verifiable)", "Sign agreement vs effective k"),
    ):
        ax.set_xticks(xpos)
        ax.set_xticklabels(K_BUCKET_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("effective k = subsequent reporting events to reference  (large ➜ small)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    # Smallest k on the right is unusual; flip so the visual story (agreement
    # falling as k shrinks) reads left-to-right.
    ax0.invert_xaxis()
    ax1.invert_xaxis()
    ax1.axhline(0.5, ls="--", color="k", lw=1, alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_category_sweep(
    sweep_by_cat: dict[str, dict[str, list[MarketAnalysis]]],
    ref_names: list[str],
    path: str,
    min_markets: int = 3,
):
    """How agreement & the payout ratio differ *by topic* across the reference
    sweep.  ``sweep_by_cat`` maps category -> reference_name -> [MarketAnalysis].
    Only categories with at least ``min_markets`` markets (at the finest
    reference) are drawn.
    """
    xpos = np.arange(len(ref_names))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Total market count per category (use max over references as the headline n).
    cats = sorted(
        sweep_by_cat,
        key=lambda c: -max((len(sweep_by_cat[c].get(r, [])) for r in ref_names),
                           default=0),
    )
    for cat in cats:
        per_ref = sweep_by_cat[cat]
        n_head = max((len(per_ref.get(r, [])) for r in ref_names), default=0)
        if n_head < min_markets:
            continue
        pear, ratio = [], []
        for r in ref_names:
            a = per_ref.get(r, [])
            pear.append(np.nanmedian([x.pearson for x in a]) if a else np.nan)
            rr = np.array([x.payout_ratio for x in a], dtype=float)
            ratio.append(np.nanmedian(rr[np.isfinite(rr)]) if np.isfinite(rr).any() else np.nan)
        label = f"{cat} (n={n_head})"
        ax0.plot(xpos, pear, "o-", label=label)
        ax1.plot(xpos, ratio, "o-", label=label)

    for ax, ylabel, title in (
        (ax0, "median Pearson corr", "Payout agreement by topic"),
        (ax1, "median total_self / total_verifiable", "Designer payout ratio by topic"),
    ):
        ax.set_xticks(xpos)
        ax.set_xticklabels(ref_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax1.axhline(1.0, ls="--", color="k", lw=1)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metric_hists(analyses: list[MarketAnalysis], path: str, title: str):
    """Distributions across markets of Pearson corr and payout ratio."""
    pear = np.array([a.pearson for a in analyses], dtype=float)
    pear = pear[~np.isnan(pear)]
    ratio = np.array([a.payout_ratio for a in analyses], dtype=float)
    ratio = ratio[np.isfinite(ratio)]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    if len(pear):
        ax0.hist(pear, bins=20, color="#1f77b4", alpha=0.8)
        ax0.axvline(np.median(pear), color="k", ls="--",
                    label=f"median = {np.median(pear):.3f}")
        ax0.legend(fontsize=8)
    ax0.set_xlabel("per-market Pearson corr (self vs verifiable payouts)")
    ax0.set_ylabel("# markets")

    if len(ratio):
        clip = np.clip(ratio, -3, 5)
        ax1.hist(clip, bins=30, color="#9467bd", alpha=0.8)
        ax1.axvline(1.0, color="k", ls="--", label="ratio = 1")
        ax1.axvline(np.median(ratio), color="#d62728", ls=":",
                    label=f"median = {np.median(ratio):.3f}")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("total_self / total_verifiable  (clipped to [-3, 5])")
    ax1.set_ylabel("# markets")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
