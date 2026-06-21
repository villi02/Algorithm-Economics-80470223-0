"""Agent definition: a signal structure + a reporting strategy.

You can also hand an agent extra ``private_info`` (an arbitrary dict) that its
strategy may read — this is the hook for "give certain information to certain
agents" beyond what their signal structure encodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .signals import SignalStructure
from .strategies import Strategy, TruthfulBayesian


@dataclass
class Agent:
    signal_structure: SignalStructure
    strategy: Strategy = field(default_factory=TruthfulBayesian)
    name: str = ""
    private_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.strategy.name}/{self.signal_structure.name}"
