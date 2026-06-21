"""Minimum informational substitutes ``k_min`` (paper Figure 2 / Theorems 1, 5).

This reproduces the *analytic* bound the paper plots in Figure 2: the smallest
number of informational substitutes ``k`` the reference agent needs so that a
focal agent's gain from deviating from truthful reporting is at most ``eps``,
under the **cross-entropy market scoring rule** (CE-MSR).

The chain of results (Section 5):

* **Theorem 1 (Eq. 3)** bounds the focal agent's adjustment term ``|Δ|`` to its
  expectation of the reference's posterior::

      |Δ| <= (1/4) * ((1-η)/η - η/(1-η)) * (1-δ)^k

  so to guarantee ``|Δ| <= eps'`` it suffices to take ::

      k >= 1/(-log(1-δ)) * log( ((1-η)/η - η/(1-η)) / (4 eps') )

* **Theorem 5 (Eq. 9)** bounds the gain from misreporting under CE-MSR by a
  function of the adjustment ``Δ`` and the *market prior* ``y = y^{(t-1)}``::

      D̂_η(Δ, y) = Δ * log( ((1-η) + Δ(η y0/y1 + (1-η)))
                            / (η - Δ(η + (1-η) y1/y0)) )

* **Remark 3**: ``D̂_η(Δ, y) = eps`` is transcendental, so we numerically solve
  ``eps' = min{ |Δ| : D̂_η(Δ, y) = eps }`` and feed it into Eq. 3 to get
  ``k_min(δ, η, eps, y1)``.

``δ`` and ``η`` are the ``(δ, η)``-informativeness parameters (Assumption 5):
``δ`` lower-bounds every agent's informativeness (``BC <= 1-δ``) and ``η``
lower-bounds the signal likelihoods (``min_{x,ω} P(x|ω) >= η``).
"""
from __future__ import annotations

import math

import numpy as np

_EPS = 1e-12


def d_hat(adj: float, y1: float, eta: float) -> float:
    """CE-MSR misreport-gain bound ``D̂_η(Δ, y)`` (paper Eq. 9).

    ``adj`` is the adjustment term ``Δ``; ``y1`` is the market prior ``y_1^{(t-1)}``.
    Returns ``+inf`` outside the domain where the log argument is positive.
    """
    y1 = float(np.clip(y1, _EPS, 1.0 - _EPS))
    y0 = 1.0 - y1
    num = (1.0 - eta) + adj * (eta * (y0 / y1) + (1.0 - eta))
    den = eta - adj * (eta + (1.0 - eta) * (y1 / y0))
    if num <= 0.0 or den <= 0.0:
        return math.inf
    return adj * math.log(num / den)


def _bisect_increasing(g, hi0: float = 1.0, maxit: int = 200):
    """Smallest root of an eventually-increasing ``g`` with ``g(0) < 0``.

    ``g`` is evaluated on ``[0, hi]``; we grow ``hi`` until ``g(hi) > 0`` then
    bisect. Returns the root location.
    """
    hi = hi0
    while g(hi) < 0.0:
        hi *= 2.0
        if hi > 1e9:  # pole is effectively unreachable; root is at the boundary
            break
    lo = 0.0
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _root_pos(eps: float, y1: float, eta: float) -> float:
    """Positive root ``Δ > 0`` of ``D̂_η(Δ, y1) = eps``.

    Substitute ``Δ = P₊(1 − e^{-u})`` where ``P₊ = η/b`` is the pole; then the
    denominator is exactly ``η·e^{-u}`` and ``log(den) = log η − u``, so the root
    (which can sit ``e^{-100}`` from the pole) is resolved in ``u``-space.
    """
    y0 = 1.0 - y1
    a = eta * (y0 / y1) + (1.0 - eta)
    b = eta + (1.0 - eta) * (y1 / y0)
    pole = eta / b
    log_eta = math.log(eta)

    def g(u: float) -> float:
        e = math.exp(-u) if u < 700.0 else 0.0
        delta = pole * (1.0 - e)
        num = (1.0 - eta) + delta * a
        # D̂ = Δ·(log num − log den), log den = log η − u
        return delta * (math.log(num) - (log_eta - u)) - eps

    u = _bisect_increasing(g)
    e = math.exp(-u) if u < 700.0 else 0.0
    return pole * (1.0 - e)


def _root_neg(eps: float, y1: float, eta: float) -> float:
    """Magnitude ``|Δ|`` of the negative root of ``D̂_η(Δ, y1) = eps``.

    Symmetric trick: ``Δ = P₋(1 − e^{-v})`` with ``P₋ = −(1−η)/a`` makes the
    numerator exactly ``(1−η)·e^{-v}``.
    """
    y0 = 1.0 - y1
    a = eta * (y0 / y1) + (1.0 - eta)
    b = eta + (1.0 - eta) * (y1 / y0)
    pole = -(1.0 - eta) / a
    log_1me = math.log(1.0 - eta)

    def g(v: float) -> float:
        e = math.exp(-v) if v < 700.0 else 0.0
        delta = pole * (1.0 - e)          # negative
        den = eta - delta * b             # > 0
        # D̂ = Δ·(log num − log den), log num = log(1−η) − v
        return delta * ((log_1me - v) - math.log(den)) - eps

    v = _bisect_increasing(g)
    e = math.exp(-v) if v < 700.0 else 0.0
    return abs(pole * (1.0 - e))


def epsilon_prime(eps: float, y1: float, eta: float) -> float:
    """``eps' = min{ |Δ| : D̂_η(Δ, y1) = eps }`` (paper Remark 3).

    ``D̂_η`` rises from 0 at ``Δ=0`` to ``+inf`` at each pole, so there is one
    root on each side of 0; we return the smaller-magnitude one.
    """
    y1 = float(np.clip(y1, _EPS, 1.0 - _EPS))
    rp = _root_pos(eps, y1, eta)
    rn = _root_neg(eps, y1, eta)
    return min(rp, rn)


def k_min(delta: float, eta: float, eps: float, y1: float) -> float:
    """Minimum informational substitutes for an ``eps``-gain guarantee (Figure 2).

    Combines Eq. 9 + Remark 3 (solve for ``eps'``) with Eq. 3 (solve for ``k``).
    Returns a real-valued ``k`` (not yet rounded up); 0 if the bound is already
    satisfied with no substitutes, NaN if ``eps'`` could not be found.
    """
    ep = epsilon_prime(eps, y1, eta)
    if not math.isfinite(ep) or ep <= 0.0:
        return float("nan")
    coef = (1.0 - eta) / eta - eta / (1.0 - eta)  # (1-η)/η - η/(1-η)
    val = coef / (4.0 * ep)
    if val <= 1.0:
        return 0.0  # bound already holds at k=0
    return math.log(val) / (-math.log(1.0 - delta))


def k_min_curve(
    delta: float, eta: float, eps: float, y1_grid: np.ndarray
) -> np.ndarray:
    """``k_min`` across a grid of market priors ``y1`` (one Figure-2 curve)."""
    return np.array([k_min(delta, eta, eps, float(y)) for y in y1_grid], dtype=float)
