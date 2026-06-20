import re
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import pandas as pd

# Canonical-name normalization for fixture team names that differ from the
# names used in results.csv / elo_history (the model's source of truth).
TEAM_NAME_MAP = {
    "USA": "United States",
    "Turkiye": "Turkey",
    "Curaçao": "Curacao",
}


def normalize_team(name: str) -> str:
    """Map a fixtures-file team name to its canonical model name."""
    return TEAM_NAME_MAP.get(str(name).strip(), str(name).strip())


@dataclass(frozen=True)
class Slot:
    """A Round-of-32 participant slot, resolved after the group stage."""
    kind: Literal["winner", "runner_up", "third"]
    group: str | None = None
    eligible: frozenset[str] | None = None


@dataclass
class Tournament:
    groups: dict[str, list[str]]
    group_matches: list[tuple[str, str, str]]
    r32_slots: list[tuple[Slot, Slot]]


_WINNER_RE = re.compile(r"^Winner\s+([A-L])$", re.I)
_RUNNER_RE = re.compile(r"^Runner-up\s+([A-L])$", re.I)
_THIRD_RE = re.compile(r"^3rd Place\s*\(([A-L/]+)\)$", re.I)


def parse_slot(text: str) -> Slot:
    """Parse an R32 slot string into a typed Slot."""
    text = str(text).strip()
    m = _WINNER_RE.match(text)
    if m:
        return Slot(kind="winner", group=m.group(1).upper())
    m = _RUNNER_RE.match(text)
    if m:
        return Slot(kind="runner_up", group=m.group(1).upper())
    m = _THIRD_RE.match(text)
    if m:
        letters = frozenset(g.strip().upper() for g in m.group(1).split("/"))
        return Slot(kind="third", eligible=letters)
    raise ValueError(f"Unrecognized R32 slot: {text!r}")


def load_tournament(path: str = "data/raw/wc2026_fixtures.csv") -> Tournament:
    """Parse the fixtures CSV into a Tournament definition."""
    df = pd.read_csv(path)
    df = df[df["stage"].astype(str) != "stage"]
    df["stage"] = df["stage"].astype(str)

    groups: dict[str, list[str]] = {}
    group_matches: list[tuple[str, str, str]] = []
    grp_rows = df[df["stage"].str.startswith("Group")]
    for stage, sub in grp_rows.groupby("stage"):
        letter = stage.split()[-1]
        teams: list[str] = []
        for _, r in sub.iterrows():
            a, b = normalize_team(r["team_a"]), normalize_team(r["team_b"])
            group_matches.append((a, b, letter))
            for t in (a, b):
                if t not in teams:
                    teams.append(t)
        groups[letter] = sorted(teams)

    r32 = df[df["stage"] == "Round of 32"]
    r32_slots = [(parse_slot(r["team_a"]), parse_slot(r["team_b"]))
                 for _, r in r32.iterrows()]

    return Tournament(groups=groups, group_matches=group_matches, r32_slots=r32_slots)


def assign_third_place(
    slot_eligibles: list[frozenset[str]],
    best_thirds: list[tuple[str, str]],
) -> list[str]:
    """
    Assign qualifying 3rd-place teams to R32 third-place slots, respecting each
    slot's eligible-group set. Returns a list of team names aligned to
    slot_eligibles. Uses backtracking (assigns the most-constrained slot first)
    so the result is deterministic. Raises ValueError if no valid assignment.
    """
    group_of = {team: g for g, team in best_thirds}
    teams = [team for _, team in best_thirds]

    order = sorted(
        range(len(slot_eligibles)),
        key=lambda i: sum(1 for t in teams if group_of[t] in slot_eligibles[i]),
    )
    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        slot_idx = order[k]
        for t in teams:
            if t in used:
                continue
            if group_of[t] in slot_eligibles[slot_idx]:
                assignment[slot_idx] = t
                used.add(t)
                if backtrack(k + 1):
                    return True
                used.discard(t)
                del assignment[slot_idx]
        return False

    if not backtrack(0):
        raise ValueError("No valid third-place assignment for given eligibilities")
    return [assignment[i] for i in range(len(slot_eligibles))]
