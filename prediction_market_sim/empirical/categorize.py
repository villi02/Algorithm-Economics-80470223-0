"""Coarse, reproducible topic classification for Polymarket markets.

Gamma's ``category`` / ``tags`` fields are empty for most markets, so we infer a
topic from the question + parent-event title using ordered keyword rules.  This is
deliberately simple and deterministic so the category breakdown is reproducible;
it is not meant to be a perfect taxonomy.  ``classify`` returns one of:

    crypto, sports, politics, economy, tech, entertainment, science, other
"""
from __future__ import annotations

import re

# Order matters: the first category whose pattern matches wins.  More specific /
# less ambiguous topics (crypto, sports) are checked before broad ones (politics).
_RULES: list[tuple[str, str]] = [
    ("crypto", r"\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|dogecoin|doge|"
               r"nft|binance|coinbase|ftx|stablecoin|altcoin|memecoin|"
               r"\$?[a-z]{3,5}coin|blockchain|satoshi|xrp|ripple|cardano)\b"),
    ("sports", r"\b(nba|nfl|mlb|nhl|ncaa|super\s?bowl|world\s?cup|champions\s?league|"
               r"premier\s?league|la\s?liga|serie\s?a|bundesliga|ufc|mma|boxing|"
               r"formula\s?1|f1|grand\s?prix|tennis|golf|olympics?|world\s?series|"
               r"playoffs?|finals?|stanley\s?cup|masters|wimbledon|"
               r"win\s+the\s+\d{4}.*\b(cup|finals?|championship|title|league))\b"),
    ("politics", r"\b(election|elections?|president(ial)?|senate|congress|governor|"
                 r"primary|primaries|ballot|parliament|prime\s?minister|democrat|"
                 r"republican|trump|biden|harris|mayor|referendum|impeach|cabinet|"
                 r"nominee|nomination|vote[ds]?|inaugurat|chancellor|"
                 r"war|ceasefire|invade|invasion|nato|nuclear|military|coup|"
                 r"shutdown|supreme\s?court|legislation|sanction)\b"),
    ("economy", r"\b(fed|federal\s?reserve|inflation|interest\s?rate|rate\s?cut|"
                r"recession|gdp|unemployment|jobs?\s?report|s&p|nasdaq|dow|"
                r"stock|stocks?|ipo|earnings|tariff|debt\s?ceiling|cpi|"
                r"price\s+of|reach\s+\$)\b"),
    ("tech", r"\b(ai|a\.i\.|gpt|chatgpt|openai|anthropic|claude|gemini|llm|tesla|"
             r"spacex|starship|twitter|\bx\b|tiktok|apple|google|microsoft|meta|"
             r"nvidia|self.?driving|robot|quantum)\b"),
    ("entertainment", r"\b(oscar|oscars|grammy|emmy|box\s?office|movie|film|album|"
                      r"spotify|netflix|taylor\s?swift|celebrity|tour|"
                      r"rotten\s?tomatoes|billboard)\b"),
    ("science", r"\b(spacecraft|nasa|mars|moon\s?landing|asteroid|covid|vaccine|"
                r"hurricane|earthquake|temperature|climate|fusion|cern)\b"),
]

_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _RULES]


def classify(text: str) -> str:
    text = text or ""
    for name, pat in _COMPILED:
        if pat.search(text):
            return name
    return "other"
