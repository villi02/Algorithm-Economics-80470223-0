"""Relative-time reference sweep: agreement vs. *percent of market life*.

The default study (``run_empirical.py``) places the reference a fixed number of
*hours* before close (``lead_1h/24h/168h``).  That mixes markets of very different
durations: 24h before a year-long market is "early", but 24h before a 3-day market
is "almost resolved".  This script instead sweeps the reference across a *fraction*
of each market's life, so the x-axis is normalised by market duration and markets
of different lengths are comparable.

For each fraction ``f`` we take the reference forecast at ``t0 + f*(t1-t0)``
(:class:`empirical.references.FractionReference`) and score every earlier price
move both ways (self-resolving CE-MSR vs. reference, verifiable log-MSR vs. the
outcome), exactly as in the main study.  We report the median across markets of
Pearson / Spearman / top-decile Jaccard and the designer payout ratio, plotted
against "% of market life before close" = ``(1 - f) * 100``.

Runs fully offline from ``empirical/cache/`` (no network) once the main study has
been run once.

    cd prediction_market_sim
    python empirical/run_fraction_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from empirical.polymarket import PolymarketClient  # noqa: E402
from empirical.references import FractionReference  # noqa: E402
from empirical.counterfactual import analyze_market  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")

# Fractions of the way through each market's life at which to place the reference.
# (1 - frac) * 100 is the "% of market life before close" used on the x-axis.
FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def run(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = PolymarketClient()
    markets = client.fetch_resolved_markets(
        n=args.n, min_volume=args.min_volume, refresh=False)
    print(f"loaded {len(markets)} markets from cache")

    refs = [FractionReference(frac=f) for f in FRACS]
    sweep: dict[float, list] = {f: [] for f in FRACS}

    used = 0
    skipped = 0
    for m in markets:
        # Offline only: skip any market whose price history isn't already cached
        # (otherwise fetch_price_history would block on the network with retries).
        cache_key = (f"hist_{m.yes_token}_{m.start_ts}_{m.fetch_end_ts}"
                     f"_f{args.fidelity}.json")
        if not os.path.exists(os.path.join(client.cache_dir, cache_key)):
            skipped += 1
            continue
        try:
            times, prices = client.fetch_price_history(
                m.yes_token, m.start_ts, m.fetch_end_ts,
                fidelity=args.fidelity, refresh=False)
        except Exception:  # noqa: BLE001
            continue
        if len(times) < args.min_reporters + 2:
            continue
        tarr, parr = np.asarray(times), np.asarray(prices)
        any_used = False
        for f, ref in zip(FRACS, refs):
            a = analyze_market(
                m.id, m.question, m.outcome, tarr, parr, ref,
                min_reporters=args.min_reporters, category=m.coarse_category)
            if a is not None:
                sweep[f].append(a)
                any_used = True
        used += int(any_used)
    print(f"analysed {used} markets ({skipped} skipped: no cached history)")

    # ----------------------------------------------------------------- aggregate
    rows = []
    for f in FRACS:
        a = sweep[f]
        if not a:
            continue
        pear = np.array([x.pearson for x in a], float)
        spear = np.array([x.spearman for x in a], float)
        jacc = np.array([x.top_decile_jaccard for x in a], float)
        ratio = np.array([x.payout_ratio for x in a], float)
        rows.append({
            "frac": f,
            "pct_before_close": round((1 - f) * 100, 1),
            "n_markets": len(a),
            "median_pearson": float(np.nanmedian(pear)),
            "median_spearman": float(np.nanmedian(spear)),
            "median_top_decile_jaccard": float(np.nanmedian(jacc)),
            "median_payout_ratio": (float(np.nanmedian(ratio[np.isfinite(ratio)]))
                                    if np.isfinite(ratio).any() else float("nan")),
        })

    csv_path = os.path.join(RESULTS_DIR, "fraction_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # ----------------------------------------------------------------- plot
    x = [r["pct_before_close"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(x, [r["median_pearson"] for r in rows], "o-", label="Pearson (payout level)")
    ax1.plot(x, [r["median_spearman"] for r in rows], "s-", label="Spearman (payout rank)")
    ax1.plot(x, [r["median_top_decile_jaccard"] for r in rows], "^-", label="top-decile Jaccard")
    ax1.set_xlabel("reference taken this % of market life before close")
    ax1.set_ylabel("median across markets")
    ax1.set_title("Agreement: self-resolving vs. verifiable payouts")
    ax1.invert_xaxis()  # closer to close (small %) on the right -> resolution
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(x, [r["median_payout_ratio"] for r in rows], "o-", color="tab:purple")
    ax2.axhline(1.0, ls="--", color="gray", lw=1)
    ax2.set_xlabel("reference taken this % of market life before close")
    ax2.set_ylabel("median total_self / total_verifiable")
    ax2.set_title("Designer total payout ratio")
    ax2.invert_xaxis()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "fig_fraction_sweep.png")
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")

    # ----------------------------------------------------------------- console
    print("\n% before close |   N  | pearson | spearman |  ratio")
    for r in rows:
        print(f"{r['pct_before_close']:>13.0f} | {r['n_markets']:>4} | "
              f"{r['median_pearson']:>7.3f} | {r['median_spearman']:>8.3f} | "
              f"{r['median_payout_ratio']:>6.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--min-volume", type=float, default=100_000.0)
    p.add_argument("--fidelity", type=int, default=60)
    p.add_argument("--min-reporters", type=int, default=5)
    run(p.parse_args())


if __name__ == "__main__":
    main()
