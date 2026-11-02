"""Accepted 1.13 differences (``oracle/accepted.yaml``): vanilla states BlueMap draws from the 1.13 client jar on
purpose, so the oracles report them as ``accepted (1.13)``, not as passes or failures.

An entry names block states and one reason of ``vanilla.REASONS``. A state is accepted only when the 1.7.10 vs 1.13
data (``oracle.vanilla``) shows that reason for it in the variant at hand (a texture the variant's BlueMap packs
replace is not a 1.13 texture there); otherwise it is judged as usual and listed as ``unbacked``.

Pattern syntax: ``minecraft:red_flower`` (every meta) or ``minecraft:red_flower[meta=0,2,4-8]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from e2e import paths
from e2e.oracle import vanilla

FILE = paths.PROJECT / "oracle" / "accepted.yaml"
LABEL = "accepted (1.13)"


@dataclass(frozen=True)
class Entry:
    states: str
    name: str
    metas: frozenset[int] | None  # None: every meta
    reason: str
    note: str

    def matches(self, name: str, meta: int) -> bool:
        return name == self.name and (self.metas is None or meta in self.metas)


def parse_pattern(pattern: str) -> tuple[str, frozenset[int] | None]:
    match = re.fullmatch(r"([a-z0-9_.-]+:[a-z0-9_./-]+)(?:\[meta=([0-9,\s-]+)\])?", pattern.strip())
    if not match:
        raise ValueError(f"accepted.yaml: bad state pattern {pattern!r}")
    if match.group(2) is None:
        return match.group(1), None
    metas: set[int] = set()
    for part in match.group(2).split(","):
        low, _, high = part.strip().partition("-")
        metas.update(range(int(low), int(high or low) + 1))
    return match.group(1), frozenset(metas)


@dataclass
class Accepted:
    entries: list[Entry]
    reasons: dict[str, str]
    vanilla_states: dict[str, dict]  # vanilla.json "states", by key
    overrides: set[str] = field(default_factory=set)
    _by_label: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_label = {f"{r['name']}:{r['meta']}": r for r in self.vanilla_states.values()}

    def entry(self, name: str, meta: int) -> Entry | None:
        return next((e for e in self.entries if e.matches(name, meta)), None)

    def check(self, name: str, meta: int) -> tuple[Entry | None, bool]:
        """The matching entry and whether the data backs its reason for this state in this variant."""
        e = self.entry(name, meta)
        if e is None:
            return None, False
        record = self._by_label.get(f"{name}:{meta}")
        return e, record is not None and e.reason in vanilla.differences(record, self.overrides)

    def describe(self, e: Entry) -> str:
        return f"{LABEL}: {self.reasons.get(e.reason, e.reason)}" + (f"; {e.note}" if e.note else "")

    def unmatched(self) -> list[str]:
        """Entries that match no vanilla state (stale patterns)."""
        return [e.states for e in self.entries
                if not any(e.matches(r["name"], r["meta"]) for r in self.vanilla_states.values())]


def load(vanilla_data: dict, overrides: set[str], path: Path = FILE) -> Accepted:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reasons = data.get("reasons", {})
    entries = []
    for raw in data.get("accepted", []):
        reason = raw["reason"]
        if reason not in vanilla.REASONS or reason not in reasons:
            raise ValueError(f"{path}: {raw['states']}: reason {reason!r} must be one of {', '.join(vanilla.REASONS)} "
                             f"and described under `reasons`")
        name, metas = parse_pattern(raw["states"])
        entries.append(Entry(raw["states"], name, metas, reason, raw.get("note", "")))
    return Accepted(entries, reasons, vanilla_data["states"], overrides)
