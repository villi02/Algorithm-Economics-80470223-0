"""Driver: fetch resolved Polymarket markets, run the counterfactual reference
sweep, and write CSVs + figures to ``empirical/results/``.

    cd prediction_market_sim
    pip install -r requirements.txt
    python empirical/run_empirical.py                 # ~200 top-volume markets
    python empirical/run_empirical.py --n 100 --min-volume 50000
    python empirical/run_empirical.py --category politics --refresh

Outputs (under empirical/results/):
    per_market.csv            one row per (market, reference); includes topic
    reference_sweep.csv       medians per reference across markets
    category_breakdown.csv    medians per (topic, reference)
    k_bucket_agreement.csv    per-reference, per-effective-k agreement stats
    fig_reference_sweep.png   how agreement / payout ratio move as r gets earlier
    fig_category_sweep.png    the same, split by inferred topic
    fig_k_bucket_agreement.png agreement vs distance-to-reference (proxy for k)
    fig_total_scatter_*.png   verifiable vs self total payout (per reference)
    fig_pooled_*.png          pooled per-reporter payouts (per reference)
    fig_hist_*.png            distribution of corr & ratio (per reference)
    fig_example_*.png         timeline panels for illustrative example markets

The headline empirical claim this supports: "On N resolved Polymarket markets, the
self-resolving mechanism's payouts track the verifiable mechanism's by <metrics>,
and the gap grows as the reference is moved from near-resolution (leakage) to
well before resolution (genuine forecast information)."
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from empirical.polymarket import PolymarketClient, MarketMeta  # noqa: E402
from empirical.references import default_reference_sweep  # noqa: E402
from empirical.counterfactual import (  # noqa: E402
    analyze_market, MarketAnalysis, series_interestingness, k_bucket_metrics,
)
from empirical import plots  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


def _pick_examples(pool, n):
    """Pick the ``n`` most illustrative markets, balanced across outcomes.

    ``pool`` is a list of ``(meta, times, prices, interestingness)``.  We sort by
    interestingness and round-robin between Yes- and No-resolved markets so the
    examples are not all the same outcome.
    """
    ranked = sorted(pool, key=lambda x: -x[3])
    yes = [x for x in ranked if x[0].outcome == 1]
    no = [x for x in ranked if x[0].outcome == 0]
    picked, i = [], 0
    while len(picked) < n and (i < len(yes) or i < len(no)):
        if i < len(yes):
            picked.append(yes[i])
        if len(picked) < n and i < len(no):
            picked.append(no[i])
        i += 1
    return picked[:n]


def run(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = PolymarketClient()

    print(f"Fetching up to {args.n} resolved markets (min_volume={args.min_volume:,.0f}"
          f"{', category=' + args.category if args.category else ''}) ...")
    markets = client.fetch_resolved_markets(
        n=args.n, min_volume=args.min_volume, category=args.category,
        refresh=args.refresh,
    )
    print(f"  got {len(markets)} markets")
    if not markets:
        print("No markets matched the filters. Try lowering --min-volume.")
        return

    references = default_reference_sweep()

    # sweep[ref_name] -> list[MarketAnalysis]; also keep per-market price series
    sweep: dict[str, list[MarketAnalysis]] = {r.name: [] for r in references}
    # sweep_by_cat[category][ref_name] -> list[MarketAnalysis]
    sweep_by_cat: dict[str, dict[str, list[MarketAnalysis]]] = {}
    per_market_rows: list[dict] = []
    # candidate (meta, times, prices, interestingness) for example timelines
    example_pool: list[tuple[MarketMeta, np.ndarray, np.ndarray, float]] = []

    for i, m in enumerate(markets):
        try:
            times, prices = client.fetch_price_history(
                m.yes_token, m.start_ts, m.fetch_end_ts,
                fidelity=args.fidelity, refresh=args.refresh,
            )
        except Exception:  # noqa: BLE001 - keep the batch going
            print(f"  [{i+1}/{len(markets)}] {m.question[:55]!r}: history fetch failed")
            traceback.print_exc(limit=1)
            continue
        if len(times) < args.min_reporters + 2:
            print(f"  [{i+1}/{len(markets)}] {m.question[:55]!r}: too few points "
                  f"({len(times)}), skipping")
            continue

        cat = m.coarse_category
        tarr, parr = np.asarray(times), np.asarray(prices)
        used = False
        for ref in references:
            a = analyze_market(
                m.id, m.question, m.outcome, tarr, parr, ref,
                min_price_change=args.min_price_change,
                min_gap_sec=args.min_gap_min * 60.0,
                min_reporters=args.min_reporters,
                category=cat,
            )
            if a is None:
                continue
            sweep[ref.name].append(a)
            sweep_by_cat.setdefault(cat, {r.name: [] for r in references})[ref.name].append(a)
            per_market_rows.append(a.to_summary_row())
            used = True
        if used:
            score = series_interestingness(tarr, parr, args.min_price_change)
            example_pool.append((m, tarr, parr, score))
        print(f"  [{i+1}/{len(markets)}] [{cat}] {m.question[:50]!r}: {len(times)} pts, "
              f"outcome={m.outcome}")

    if not per_market_rows:
        print("No analyzable markets after filtering. Try --min-reporters 3.")
        return

    # ----------------------------------------------------------------- CSVs
    header = list(per_market_rows[0].keys())
    _write_csv(os.path.join(RESULTS_DIR, "per_market.csv"), header,
               [[row[h] for h in header] for row in per_market_rows])

    sweep_rows = []
    for ref in references:
        a = sweep[ref.name]
        if not a:
            continue
        pear = np.array([x.pearson for x in a], dtype=float)
        spear = np.array([x.spearman for x in a], dtype=float)
        jacc = np.array([x.top_decile_jaccard for x in a], dtype=float)
        ratio = np.array([x.payout_ratio for x in a], dtype=float)
        argmax = np.array([x.argmax_match for x in a], dtype=float)
        sweep_rows.append([
            ref.name, len(a),
            round(np.nanmedian(pear), 4), round(np.nanmedian(spear), 4),
            round(np.nanmedian(jacc), 4),
            round(float(np.nanmedian(ratio[np.isfinite(ratio)])) if np.isfinite(ratio).any() else float("nan"), 4),
            round(float(np.mean(argmax)), 4),
        ])
    _write_csv(
        os.path.join(RESULTS_DIR, "reference_sweep.csv"),
        ["reference", "n_markets", "median_pearson", "median_spearman",
         "median_top_decile_jaccard", "median_payout_ratio", "argmax_match_rate"],
        sweep_rows,
    )

    # ----------------------------------------------------- category breakdown
    cat_rows = []
    for cat in sorted(sweep_by_cat):
        for ref in references:
            a = sweep_by_cat[cat].get(ref.name, [])
            if not a:
                continue
            ratio = np.array([x.payout_ratio for x in a], dtype=float)
            cat_rows.append([
                cat, ref.name, len(a),
                round(float(np.nanmedian([x.pearson for x in a])), 4),
                round(float(np.nanmedian([x.spearman for x in a])), 4),
                round(float(np.nanmedian(ratio[np.isfinite(ratio)]))
                      if np.isfinite(ratio).any() else float("nan"), 4),
                round(float(np.mean([x.argmax_match for x in a])), 4),
            ])
    _write_csv(
        os.path.join(RESULTS_DIR, "category_breakdown.csv"),
        ["category", "reference", "n_markets", "median_pearson",
         "median_spearman", "median_payout_ratio", "argmax_match_rate"],
        cat_rows,
    )

    # ------------------------------ per-reporter k (distance to reference)
    k_rows = []
    for ref in references:
        a = sweep[ref.name]
        if not a:
            continue
        for row in k_bucket_metrics(a):
            k_rows.append([
                ref.name, row["k_label"], row["n"],
                round(row["pearson"], 4) if not np.isnan(row["pearson"]) else "",
                round(row["sign_agree"], 4) if not np.isnan(row["sign_agree"]) else "",
                round(row["mean_abs_diff"], 5),
            ])
    _write_csv(
        os.path.join(RESULTS_DIR, "k_bucket_agreement.csv"),
        ["reference", "k_bucket", "n_transitions", "pearson",
         "sign_agree_rate", "mean_abs_payout_diff"],
        k_rows,
    )

    # ----------------------------------------------------------------- figures
    plots.plot_reference_sweep(sweep, os.path.join(RESULTS_DIR, "fig_reference_sweep.png"))
    print(f"  wrote {os.path.join(RESULTS_DIR, 'fig_reference_sweep.png')}")

    ref_names = [r.name for r in references]
    plots.plot_category_sweep(
        sweep_by_cat, ref_names,
        os.path.join(RESULTS_DIR, "fig_category_sweep.png"))
    print(f"  wrote {os.path.join(RESULTS_DIR, 'fig_category_sweep.png')}")

    plots.plot_k_bucket_agreement(
        sweep, ref_names,
        os.path.join(RESULTS_DIR, "fig_k_bucket_agreement.png"))
    print(f"  wrote {os.path.join(RESULTS_DIR, 'fig_k_bucket_agreement.png')}")

    for ref in references:
        a = sweep[ref.name]
        if len(a) < 2:
            continue
        plots.plot_total_scatter(
            a, os.path.join(RESULTS_DIR, f"fig_total_scatter_{ref.name}.png"),
            f"Total payout per market — reference = {ref.name}")
        plots.plot_pooled_transitions(
            a, os.path.join(RESULTS_DIR, f"fig_pooled_{ref.name}.png"),
            f"Per-reporter payouts (pooled) — reference = {ref.name}")
        plots.plot_metric_hists(
            a, os.path.join(RESULTS_DIR, f"fig_hist_{ref.name}.png"),
            f"Across-market distributions — reference = {ref.name}")
    print(f"  wrote per-reference scatter / pooled / hist figures")

    # Example timelines: the most *illustrative* markets (price travels through the
    # middle, not a flat near-0/1 line), balanced across the two outcomes.
    ref_for_examples = references[1]  # 24h lead
    example_markets = _pick_examples(example_pool, args.n_examples)
    n_drawn = 0
    for m, times, prices, _ in example_markets:
        a = analyze_market(m.id, m.question, m.outcome, times, prices,
                           ref_for_examples, min_price_change=args.min_price_change,
                           min_gap_sec=args.min_gap_min * 60.0,
                           min_reporters=args.min_reporters,
                           category=m.coarse_category)
        if a is None:
            continue
        safe = "".join(c if c.isalnum() else "_" for c in m.slug or m.id)[:40]
        plots.plot_market_timeline(
            a, a.ref_price,
            os.path.join(RESULTS_DIR, f"fig_example_{safe}.png"))
        n_drawn += 1
    print(f"  wrote {n_drawn} example timelines")

    # ----------------------------------------------------------------- console
    _print_headline(sweep, references)
    _print_category_headline(sweep_by_cat, references)


def _print_headline(sweep, references):
    print("\n" + "=" * 70)
    print("HEADLINE RESULTS (median across markets)")
    print("=" * 70)
    print(f"{'reference':<16}{'N':>4}{'pearson':>10}{'spearman':>10}"
          f"{'ratio':>9}{'argmax%':>9}")
    for ref in references:
        a = sweep[ref.name]
        if not a:
            continue
        pear = np.nanmedian([x.pearson for x in a])
        spear = np.nanmedian([x.spearman for x in a])
        ratio = np.array([x.payout_ratio for x in a], dtype=float)
        ratio_m = np.nanmedian(ratio[np.isfinite(ratio)]) if np.isfinite(ratio).any() else float("nan")
        argmax = np.mean([x.argmax_match for x in a])
        print(f"{ref.name:<16}{len(a):>4}{pear:>10.3f}{spear:>10.3f}"
              f"{ratio_m:>9.3f}{argmax*100:>8.1f}%")
    print("=" * 70)
    print("Read: as the reference moves from near-resolution (lead_1h) to well\n"
          "before it (life_50pct), agreement with the verifiable scheme should\n"
          "fall — isolating outcome leakage from genuine forecast information.")


def _print_category_headline(sweep_by_cat, references):
    """Topic breakdown at the conservative (substantially-pre-resolution) reference."""
    ref = next((r for r in references if r.name == "life_50pct"), references[-1])
    print(f"\nBY TOPIC at reference = {ref.name} (median across markets)")
    print(f"{'topic':<16}{'N':>4}{'pearson':>10}{'spearman':>10}{'ratio':>9}")
    rows = []
    for cat, per_ref in sweep_by_cat.items():
        a = per_ref.get(ref.name, [])
        if not a:
            continue
        ratio = np.array([x.payout_ratio for x in a], dtype=float)
        rows.append((
            len(a), cat,
            np.nanmedian([x.pearson for x in a]),
            np.nanmedian([x.spearman for x in a]),
            np.nanmedian(ratio[np.isfinite(ratio)]) if np.isfinite(ratio).any() else float("nan"),
        ))
    for n, cat, pear, spear, ratio_m in sorted(rows, reverse=True):
        print(f"{cat:<16}{n:>4}{pear:>10.3f}{spear:>10.3f}{ratio_m:>9.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="number of markets to analyse")
    p.add_argument("--min-volume", type=float, default=100_000.0,
                   help="minimum market volume (USD)")
    p.add_argument("--category", type=str, default=None,
                   help="restrict to a Gamma category (e.g. politics)")
    p.add_argument("--fidelity", type=int, default=60,
                   help="price sampling resolution in minutes")
    p.add_argument("--min-price-change", type=float, default=0.0,
                   help="min |Δprice| for a snapshot to count as a reporting event")
    p.add_argument("--min-gap-min", type=float, default=0.0,
                   help="min minutes between reporting events")
    p.add_argument("--min-reporters", type=int, default=5,
                   help="skip a (market, reference) with fewer reporters")
    p.add_argument("--n-examples", type=int, default=6,
                   help="number of example timeline figures to draw")
    p.add_argument("--refresh", action="store_true",
                   help="ignore cache and re-fetch from the API")
    run(p.parse_args())


if __name__ == "__main__":
    main()
