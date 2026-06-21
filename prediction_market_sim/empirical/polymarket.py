"""Polymarket API client (Gamma metadata + CLOB price history) with disk caching.

Two public endpoints are used, no API key required:

* **Gamma** ``https://gamma-api.polymarket.com/markets`` — market metadata,
  including the resolved ``outcomePrices`` and the CLOB ``clobTokenIds``.
* **CLOB** ``https://clob.polymarket.com/prices-history`` — the historical
  midpoint price series for a single outcome token.

The CLOB endpoint caps each response at a few hundred points, so
:meth:`PolymarketClient.fetch_price_history` chunks a long date range into
windows small enough to come back fully, then concatenates and de-duplicates.

All responses are cached as JSON under ``empirical/cache/`` so repeated runs are
offline and fast.  Delete that directory to force a refresh.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable

import requests

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "cache")


def _iso_to_unix(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip()
    # Gamma uses both "...Z" and "+00" / "+00:00" suffixes.
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    try:
        s2 = s.replace(" ", "T")
        if s2.endswith("+00"):
            s2 = s2[:-3] + "+00:00"
        return int(datetime.fromisoformat(s2).timestamp())
    except ValueError:
        return None


def _loads_maybe(value):
    """Several Gamma fields are JSON *strings* (e.g. '["Yes", "No"]')."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@dataclass
class MarketMeta:
    """A resolved, binary Yes/No market with a clean 0/1 outcome."""

    id: str
    question: str
    slug: str
    condition_id: str
    yes_token: str          # CLOB token id for the "Yes" outcome
    outcome: int            # realised Y: 1 if "Yes" resolved true, else 0
    start_ts: int           # market start (unix seconds, UTC)
    end_ts: int             # market end / endDate (unix seconds, UTC)
    volume: float
    category: str = ""
    resolve_ts: int = 0     # closedTime (settlement); >= end_ts when known
    event_title: str = ""   # parent event title (extra text for categorisation)

    @property
    def fetch_end_ts(self) -> int:
        """Upper bound for price-history fetch: covers any trading past endDate."""
        return max(self.end_ts, self.resolve_ts)

    @property
    def coarse_category(self) -> str:
        """Keyword-based topic label (politics / sports / crypto / ...).

        Gamma's own ``category``/``tags`` are empty for most markets, so we infer
        a coarse, reproducible topic from the question + parent event title.
        """
        from .categorize import classify
        return self.category.strip().lower() or classify(
            f"{self.question} {self.event_title}")

    @property
    def duration_days(self) -> float:
        return (self.end_ts - self.start_ts) / 86400.0


class PolymarketClient:
    def __init__(self, cache_dir: str = CACHE_DIR, request_pause: float = 0.3,
                 timeout: float = 30.0, max_retries: int = 5):
        self.cache_dir = cache_dir
        self.request_pause = request_pause
        self.timeout = timeout
        self.max_retries = max_retries
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "srpm-empirical/1.0"})

    # ------------------------------------------------------------------ caching
    def _cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, name)

    def _get_json(self, url: str, params: dict):
        """GET with retry + exponential backoff for rate-limits / transient errors.

        Retries on HTTP 429/5xx and on connection/timeout errors (which is what a
        long fetch burst produces).  Honours a ``Retry-After`` header when present.
        Raises the last error if all retries are exhausted.
        """
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            time.sleep(self.request_pause)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = float(resp.headers.get("Retry-After", backoff))
                    time.sleep(min(wait, 30.0))
                    backoff = min(backoff * 2, 30.0)
                    last_exc = requests.HTTPError(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
                last_exc = exc
                time.sleep(min(backoff, 30.0))
                backoff = min(backoff * 2, 30.0)
        if last_exc is not None:
            raise last_exc
        raise requests.RequestException("request failed after retries")

    # ------------------------------------------------------------ market lookup
    def fetch_resolved_markets(
        self,
        n: int = 50,
        min_volume: float = 100_000.0,
        category: str | None = None,
        page_size: int = 100,
        max_pages: int = 60,
        refresh: bool = False,
    ) -> list[MarketMeta]:
        """Return up to ``n`` resolved binary Yes/No markets, highest volume first.

        Markets are kept only if they (1) are closed, (2) have exactly the two
        outcomes Yes/No, (3) resolved cleanly to 0/1, (4) expose two CLOB token
        ids, and (5) clear ``min_volume``.
        """
        key = f"markets_n{n}_v{int(min_volume)}_{category or 'all'}.json"
        path = self._cache_path(key)
        if os.path.exists(path) and not refresh:
            raw = json.load(open(path))
            return [MarketMeta(**m) for m in raw]

        kept: list[MarketMeta] = []
        offset = 0
        for _ in range(max_pages):
            if len(kept) >= n:
                break
            params = {
                "closed": "true",
                "limit": page_size,
                "offset": offset,
                "order": "volumeNum",
                "ascending": "false",
            }
            batch = self._get_json(GAMMA_URL, params)
            if not batch:
                break
            offset += len(batch)
            for m in batch:
                meta = self._parse_market(m, min_volume, category)
                if meta is not None:
                    kept.append(meta)
                    if len(kept) >= n:
                        break

        json.dump([asdict(m) for m in kept], open(path, "w"), indent=1)
        return kept

    @staticmethod
    def _parse_market(m: dict, min_volume: float, category: str | None) -> MarketMeta | None:
        if not m.get("closed"):
            return None
        vol = float(m.get("volumeNum") or 0.0)
        if vol < min_volume:
            return None
        if category and (m.get("category") or "").lower() != category.lower():
            return None

        outcomes = _loads_maybe(m.get("outcomes"))
        if not (isinstance(outcomes, list) and len(outcomes) == 2):
            return None
        if [str(o).lower() for o in outcomes] != ["yes", "no"]:
            return None

        prices = _loads_maybe(m.get("outcomePrices"))
        tokens = _loads_maybe(m.get("clobTokenIds"))
        if not (isinstance(prices, list) and len(prices) == 2):
            return None
        if not (isinstance(tokens, list) and len(tokens) == 2):
            return None
        try:
            yes_price = float(prices[0])
        except (TypeError, ValueError):
            return None
        # Require a clean resolution (~0 or ~1), not an unresolved / 50-50 row.
        if yes_price > 0.99:
            outcome = 1
        elif yes_price < 0.01:
            outcome = 0
        else:
            return None

        start_ts = _iso_to_unix(m.get("startDate"))
        closed_ts = _iso_to_unix(m.get("closedTime"))
        end_ts = _iso_to_unix(m.get("endDate")) or closed_ts
        if start_ts is None or end_ts is None or end_ts <= start_ts:
            return None
        resolve_ts = max(end_ts, closed_ts or 0)

        events = m.get("events") or []
        event_title = str(events[0].get("title", "")) if events else ""

        return MarketMeta(
            id=str(m.get("id")),
            question=str(m.get("question", "")),
            slug=str(m.get("slug", "")),
            condition_id=str(m.get("conditionId", "")),
            yes_token=str(tokens[0]),
            outcome=outcome,
            start_ts=start_ts,
            end_ts=end_ts,
            volume=vol,
            category=str(m.get("category") or ""),
            resolve_ts=resolve_ts,
            event_title=event_title,
        )

    # ------------------------------------------------------------ price history
    def fetch_price_history(
        self,
        token: str,
        start_ts: int,
        end_ts: int,
        fidelity: int = 60,
        target_points: int = 250,
        refresh: bool = False,
    ) -> tuple[list[int], list[float]]:
        """Full Yes-price series for ``token`` over ``[start_ts, end_ts]``.

        ``fidelity`` is the sampling resolution in minutes.  The range is fetched
        in chunks of ``target_points * fidelity`` minutes (the CLOB endpoint
        returns nothing if a single request would exceed a few hundred points).
        Returns ``(times, prices)`` sorted ascending with consecutive duplicate
        timestamps removed.
        """
        key = f"hist_{token}_{start_ts}_{end_ts}_f{fidelity}.json"
        path = self._cache_path(key)
        if os.path.exists(path) and not refresh:
            d = json.load(open(path))
            return d["t"], d["p"]

        chunk = max(1, target_points) * fidelity * 60  # seconds per request
        merged: dict[int, float] = {}
        lo = start_ts
        while lo < end_ts:
            hi = min(lo + chunk, end_ts)
            try:
                data = self._get_json(
                    CLOB_HISTORY_URL,
                    {"market": token, "startTs": lo, "endTs": hi, "fidelity": fidelity},
                )
                for pt in data.get("history", []):
                    merged[int(pt["t"])] = float(pt["p"])
            except requests.RequestException:
                pass  # skip a bad window, keep going
            lo = hi

        # Fallback: some long-lived markets only answer the interval=max form.
        if not merged:
            for fid in (fidelity, 180, 360, 720, 1440):
                try:
                    data = self._get_json(
                        CLOB_HISTORY_URL,
                        {"market": token, "interval": "max", "fidelity": fid},
                    )
                except requests.RequestException:
                    continue
                if data.get("history"):
                    for pt in data["history"]:
                        merged[int(pt["t"])] = float(pt["p"])
                    break

        times = sorted(merged)
        prices = [merged[t] for t in times]
        # Only cache non-empty results, so a transient failure is retried next run.
        if times:
            json.dump({"t": times, "p": prices}, open(path, "w"))
        return times, prices
