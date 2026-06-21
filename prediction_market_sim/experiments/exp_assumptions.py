"""Single-assumption stress tests (prototype): A1 (rationality) and A2 (common prior).

The simulator makes the paper's assumptions A1-A4 hold by construction, and RQ1 already
varies A5. This experiment violates ONE assumption at a time, holding everything else
fixed, and watches the market's terminal *reference* drift from the ideal
full-information posterior. That gap (``mean_abs_gap_to_full_info``) is the quantity the
whole self-resolving idea depends on: if the reference is a good proxy for the truth the
gap is ~0 and the reference matches the ideal full-information posterior. (We measure the
aggregation gap and accuracy here, not payout agreement -- that is RQ2's question.)

Both knobs are calibrated so that the *baseline* value recovers the assumption:

* **A1 (Bayesian-rationality component).** Every agent is a *boundedly rational* Bayesian:
  it adds zero-mean Gaussian noise of scale ``sigma`` (in log-odds) to its truthful report.
  This perturbs the Bayesian-rationality part of A1, not risk neutrality. ``sigma = 0`` is
  the exact-Bayesian baseline. In this cumulative reporting model per-agent errors are not
  averaged away but random-walk into the terminal reference, so the gap grows with ``sigma``.

* **A2 (common prior).** The world draws ``Y`` under the true prior, but the market anchors
  on a *mis-specified* common prior. Agents still share a single common prior; it simply
  disagrees with the data-generating prior by a logit gap ``d`` (so ``d = 0`` is the
  correct-prior baseline). A wrong prior shifts the terminal log-odds by a constant, biasing
  the aggregate, most where the signals are too weak to overrule it.

This is a deliberately cheap prototype (it reuses ``SelfResolvingMarket`` +
``aggregation_stats`` with no changes to the core), to see whether the single-assumption
story is paper-worthy before tackling the harder A3/A4 violations.
"""
from __future__ import annotations

import numpy as np

from _common import save_fig, write_csv
import matplotlib.pyplot as plt

from srpm import (
    Agent,
    MarketConfig,
    NoisyTruthful,
    SelfResolvingMarket,
    Strategy,
    TruthfulBayesian,
    aggregation_stats,
    belief_from_logodds,
    binary_symmetric,
    logit,
)

SEED = 11
N_RUNS = 8000
PRIOR = 0.5
N = 8
Q = 0.6

# A1: noise scale (log-odds) injected into every agent's truthful report.
A1_SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
# A2: logit-disagreement between the market's assumed prior and the true prior.
A2_DISAGREEMENTS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


class SubjectivePrior(Strategy):
    """Anchor the running estimate on a (possibly wrong) common prior.

    Strips the true-prior seed out of the previous report and substitutes the assumed
    prior, then adds the agent's own signal. Placed at the head of the market, it makes
    the whole market behave as if it shared this (wrong) common prior, which then
    persists additively through the truthful agents that follow.
    """

    def __init__(self, assumed_p1: float, true_p1: float):
        self.assumed = float(assumed_p1)
        self.true = float(true_p1)
        self.name = f"subjprior(p={assumed_p1:.3f})"

    def report(self, ctx) -> np.ndarray:
        rebased = ctx.prev_logodds - logit(self.true) + logit(self.assumed)
        return belief_from_logodds(rebased + ctx.own_llr)


def _stats_for(agents, rng):
    market = SelfResolvingMarket(MarketConfig(prior_p1=PRIOR, k=1))
    return aggregation_stats(market.run_many(agents, N_RUNS, rng))


def sweep_a1(rng):
    rows = []
    for sigma in A1_SIGMAS:
        agents = [Agent(binary_symmetric(Q), NoisyTruthful(sigma)) for _ in range(N)]
        st = _stats_for(agents, rng)
        rows.append(["A1_rationality", "noise_sigma", sigma, st.accuracy,
                     st.mean_brier, st.mean_abs_gap_to_full_info, st.mean_total_cost])
    return rows


def sweep_a2(rng):
    rows = []
    base = logit(PRIOR)
    for d in A2_DISAGREEMENTS:
        assumed_p1 = float(1.0 / (1.0 + np.exp(-(base + d))))  # sigmoid(logit(prior)+d)
        agents = [Agent(binary_symmetric(Q), SubjectivePrior(assumed_p1, PRIOR))]
        agents += [Agent(binary_symmetric(Q), TruthfulBayesian()) for _ in range(N - 1)]
        st = _stats_for(agents, rng)
        rows.append(["A2_common_prior", "logit_disagreement", d, st.accuracy,
                     st.mean_brier, st.mean_abs_gap_to_full_info, st.mean_total_cost])
    return rows


def _panel(ax, xs, gaps, accs, xlabel, title):
    ax.plot(xs, gaps, marker="o", color="tab:red", label="gap to full-info")
    ax.set(xlabel=xlabel, ylabel="|market - full-info posterior|", title=title)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xs, accs, marker="s", ls="--", color="tab:blue", label="accuracy")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.45, 1.02)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)


def main() -> None:
    # Independent RNG streams per sweep (decoupled but reproducible from one seed), so the
    # A2 numbers do not depend on how many draws A1 happened to consume.
    rng_a1, rng_a2 = np.random.default_rng(SEED).spawn(2)
    a1 = sweep_a1(rng_a1)
    a2 = sweep_a2(rng_a2)
    rows = a1 + a2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _panel(ax1, A1_SIGMAS, [r[5] for r in a1], [r[3] for r in a1],
           r"per-agent report noise $\sigma$ (0 = exact Bayesian)",
           "A1: bounded rationality")
    _panel(ax2, A2_DISAGREEMENTS, [r[5] for r in a2], [r[3] for r in a2],
           r"prior misspecification $d$ (0 = correct prior)",
           "A2: mis-specified common prior")

    save_fig(fig, "exp_assumptions.png")
    write_csv("exp_assumptions.csv",
              ["assumption", "knob", "knob_value", "accuracy", "brier",
               "gap_to_full_info", "mean_total_cost"], rows)

    print("A1 (rationality) -- gap & accuracy vs noise sigma:")
    for r in a1:
        print(f"  sigma={r[2]:.2f}  gap={r[5]:.4f}  acc={r[3]:.3f}  cost={r[6]:.3f}")
    print("A2 (common prior) -- gap & accuracy vs logit-disagreement:")
    for r in a2:
        print(f"  d={r[2]:.2f}  gap={r[5]:.4f}  acc={r[3]:.3f}  cost={r[6]:.3f}")


if __name__ == "__main__":
    main()
