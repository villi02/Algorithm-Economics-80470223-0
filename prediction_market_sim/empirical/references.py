"""Choosing the reference ``r`` against which counterfactual payouts are scored.

In the paper the reference is the *terminal* agent ``T`` — the last, best-informed
forecast.  On observational data the analogue is a *late* market price.  But "late"
hides two distinct channels of informativeness:

* **Outcome leakage** — a price 1 hour before resolution is close to the realised
  0/1 outcome simply because the question is nearly settled.  Using it as ``r`` is
  an (optimistic) upper bound on how well the mechanism can track ground truth.
* **Genuine forecast information** — a price a week before resolution reflects
  aggregated belief that is well-informed but *not* omniscient.  This is closest
  to the paper's spirit and is the most honest basis for the empirical claim.

A :class:`Reference` maps a market's event series to ``(n_reporters, ref_price)``:
the first ``n_reporters`` events are the sequential "reporters" being scored, and
``ref_price`` is the Yes-probability of the reference forecast they are scored
against.  Sweeping the reference from very-late to substantially-before-resolution
lets us separate the two channels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Reference:
    """Base class. Subclasses implement :meth:`resolve`."""

    name: str = "reference"

    def resolve(self, times: np.ndarray, prices: np.ndarray) -> tuple[int, float]:
        raise NotImplementedError


@dataclass
class LeadReference(Reference):
    """``r`` = the forecast a fixed *lead time* before the market's last point.

    Reporters are every event strictly earlier than that reference event.  Small
    ``lead_hours`` (e.g. 1) ⇒ near-final price (leakage upper bound); large
    ``lead_hours`` (e.g. 168) ⇒ an earlier, less omniscient forecast.
    """

    lead_hours: float = 24.0

    def __post_init__(self):
        if self.name == "reference":
            self.name = f"lead_{int(self.lead_hours)}h"

    def resolve(self, times, prices):
        t_end = times[-1]
        cutoff = t_end - self.lead_hours * 3600.0
        # The reference forecast = last event at or before the cutoff.
        idx = int(np.searchsorted(times, cutoff, side="right") - 1)
        if idx < 0:
            idx = 0
        return idx, float(prices[idx])


@dataclass
class FractionReference(Reference):
    """``r`` = the forecast at a fraction of the way through the market's life.

    ``frac=0.5`` ⇒ the midpoint forecast: well-informed but far from resolution,
    the most conservative (paper-spirit) reference.
    """

    frac: float = 0.5

    def __post_init__(self):
        if self.name == "reference":
            self.name = f"life_{int(self.frac * 100)}pct"

    def resolve(self, times, prices):
        t0, t1 = times[0], times[-1]
        cutoff = t0 + self.frac * (t1 - t0)
        idx = int(np.searchsorted(times, cutoff, side="right") - 1)
        if idx < 0:
            idx = 0
        return idx, float(prices[idx])


@dataclass
class WindowReference(Reference):
    """``r`` = the *average* Yes-price over a late window ``[lo_hours, hi_hours]``
    before the last point (paper Section 6.2: averaging several reference agents
    reduces payoff variance).  Reporters are events before the window starts.
    """

    lo_hours: float = 1.0
    hi_hours: float = 24.0

    def __post_init__(self):
        if self.name == "reference":
            self.name = f"window_{int(self.lo_hours)}-{int(self.hi_hours)}h"

    def resolve(self, times, prices):
        t_end = times[-1]
        win_lo = t_end - self.hi_hours * 3600.0   # earliest time in the window
        win_hi = t_end - self.lo_hours * 3600.0   # latest time in the window
        in_window = (times >= win_lo) & (times <= win_hi)
        n_reporters = int(np.searchsorted(times, win_lo, side="left"))
        if in_window.any():
            ref_price = float(np.mean(prices[in_window]))
        else:
            idx = max(0, n_reporters - 1)
            ref_price = float(prices[idx])
        return n_reporters, ref_price


def default_reference_sweep() -> list[Reference]:
    """The references analysed by default, ordered from most-leakage to least.

    1h lead   – upper bound (outcome leakage dominates)
    24h lead  – moderately late midpoint
    1-week    – substantially before resolution (paper's spirit)
    50%-life  – the midpoint forecast (most conservative)
    1–24h avg – variance-reduced late window (Section 6.2)
    """
    return [
        LeadReference(lead_hours=1),
        LeadReference(lead_hours=24),
        LeadReference(lead_hours=168),
        FractionReference(frac=0.5),
        WindowReference(lo_hours=1, hi_hours=24),
    ]
