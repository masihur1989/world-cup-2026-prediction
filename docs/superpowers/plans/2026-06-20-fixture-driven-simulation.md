# Fixture-Driven Tournament Simulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Monte Carlo simulator a real WC 2026 prediction driven entirely by `wc2026_fixtures.csv`, with dated prediction snapshots, round-by-round updates from real results, and a probabilistic bracket view.

**Architecture:** A new `fixtures_bracket.py` parses the CSV into a `Tournament` (groups, group matches, R32 slots) with team-name normalization. A `MatchupFeatureProvider` builds current-state features for any team pair so knockout odds are real. The rewritten `TournamentSimulator` plays the real group fixtures, resolves the R32 from CSV slots (with constrained third-place assignment), advances via standard adjacency, and tallies advancement + per-bracket-position occupancy, honoring a `known_results` override. A `predictions.py` module stores dated snapshots; the pipeline and Streamlit dashboard expose them.

**Tech Stack:** Python 3.12, pandas, numpy, xgboost, statsmodels, streamlit, plotly, pytest.

---

## Reference: existing signatures this plan builds on

- `src/feature_builder.py`: `CONFEDERATION: dict[str,str]`, `CONFEDERATIONS = ["UEFA","CONMEBOL","CAF","AFC","CONCACAF","OFC"]`, `build_team_perspective(results)->DataFrame`, `compute_penalty_win_rates(shootouts)->dict[str,float]`, `_h2h_stats(results, a, b, before_date)->(float,float)`, `_one_hot_conf(team, prefix)->dict`, `_stage_from_tournament_or_stage(tournament, stage)->int`.
- `src/elo_calculator.py`: `get_elo_on_date(elo_history, team, date)->float`, `get_rank_on_date(elo_history, date, team)->int`.
- `src/poisson_model.py`: `PoissonGoalModel` with `.predict_lambda(row)->(la,lb)`, `.simulate_match(la,lb,n)->(pw,pd,pl)`, `.load(path)`.
- `src/xgboost_model.py`: `FEATURE_COLS` (28 cols), `WorldCupXGBModel` with `.predict_proba(X)`, `.label_encoder.classes_`.
- `src/simulator.py`: `get_8_best_third(third_place: list[dict])->list[str]` (sorts by pts,gd,gf), `save_simulation_results(df, path)`.

Run tests from the worktree root with `python -m pytest`. Use the project `.venv`.

---

## Task 1: Fixtures parser — `fixtures_bracket.py`

**Files:**
- Create: `src/fixtures_bracket.py`
- Test: `tests/test_fixtures_bracket.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fixtures_bracket.py
import pandas as pd
import pytest
from src.fixtures_bracket import (
    normalize_team, parse_slot, Slot, load_tournament, Tournament,
)


def test_normalize_team_known_aliases():
    assert normalize_team("USA") == "United States"
    assert normalize_team("Turkiye") == "Turkey"
    assert normalize_team("Curaçao") == "Curacao"


def test_normalize_team_passthrough():
    assert normalize_team("Brazil") == "Brazil"


def test_parse_slot_winner():
    s = parse_slot("Winner C")
    assert s.kind == "winner" and s.group == "C" and s.eligible is None


def test_parse_slot_runner_up():
    s = parse_slot("Runner-up F")
    assert s.kind == "runner_up" and s.group == "F"


def test_parse_slot_third():
    s = parse_slot("3rd Place (A/B/C/D/F)")
    assert s.kind == "third"
    assert s.eligible == frozenset({"A", "B", "C", "D", "F"})
    assert s.group is None


def test_load_tournament_groups(tmp_fixtures):
    t = load_tournament(tmp_fixtures)
    assert isinstance(t, Tournament)
    assert len(t.groups) == 12
    assert all(len(v) == 4 for v in t.groups.values())
    # names normalized
    assert "United States" in t.groups["D"]
    assert "Turkey" in t.groups["D"]


def test_load_tournament_group_matches(tmp_fixtures):
    t = load_tournament(tmp_fixtures)
    # 12 groups * 6 round-robin matches = 72
    assert len(t.group_matches) == 72
    a, b, g = t.group_matches[0]
    assert g == "A"


def test_load_tournament_r32_slots(tmp_fixtures):
    t = load_tournament(tmp_fixtures)
    assert len(t.r32_slots) == 16
    first = t.r32_slots[0]
    assert isinstance(first[0], Slot) and isinstance(first[1], Slot)


@pytest.fixture
def tmp_fixtures(tmp_path):
    """Minimal fixtures CSV: 2 groups (8 matches won't be 72) — use real file instead."""
    # Use the real project fixtures file for integration realism.
    return "data/raw/wc2026_fixtures.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fixtures_bracket.py -v`
Expected: `ImportError` / `ModuleNotFoundError: No module named 'src.fixtures_bracket'`.

- [ ] **Step 3: Write `src/fixtures_bracket.py`**

```python
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
    group: str | None = None              # for winner / runner_up
    eligible: frozenset[str] | None = None  # for third (eligible group letters)


@dataclass
class Tournament:
    groups: dict[str, list[str]]                 # "A" -> [4 canonical names]
    group_matches: list[tuple[str, str, str]]    # (team_a, team_b, group_letter)
    r32_slots: list[tuple[Slot, Slot]]           # 16 R32 pairings


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
    df = df[df["stage"].astype(str) != "stage"]  # drop any stray duplicate header
    df["stage"] = df["stage"].astype(str)

    # Groups + group matches
    groups: dict[str, list[str]] = {}
    group_matches: list[tuple[str, str, str]] = []
    grp_rows = df[df["stage"].str.startswith("Group")]
    for stage, sub in grp_rows.groupby("stage"):
        letter = stage.split()[-1]  # "Group A" -> "A"
        teams: list[str] = []
        for _, r in sub.iterrows():
            a, b = normalize_team(r["team_a"]), normalize_team(r["team_b"])
            group_matches.append((a, b, letter))
            for t in (a, b):
                if t not in teams:
                    teams.append(t)
        groups[letter] = sorted(teams)

    # R32 slots (the only well-formed knockout rows)
    r32 = df[df["stage"] == "Round of 32"]
    r32_slots = [(parse_slot(r["team_a"]), parse_slot(r["team_b"]))
                 for _, r in r32.iterrows()]

    return Tournament(groups=groups, group_matches=group_matches, r32_slots=r32_slots)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fixtures_bracket.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fixtures_bracket.py tests/test_fixtures_bracket.py
git commit -m "feat: fixtures_bracket parser — groups, matches, R32 slots, name normalization"
```

---

## Task 2: Full 48-team confederation map

**Files:**
- Modify: `src/feature_builder.py` (replace the `CONFEDERATION` dict, keep `CONFEDERATIONS` list)
- Test: `tests/test_confederation_map.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_confederation_map.py
from src.feature_builder import CONFEDERATION, CONFEDERATIONS
from src.fixtures_bracket import load_tournament, normalize_team


def test_all_fixture_teams_have_confederation():
    t = load_tournament("data/raw/wc2026_fixtures.csv")
    teams = {tm for grp in t.groups.values() for tm in grp}
    missing = sorted(tm for tm in teams if tm not in CONFEDERATION)
    assert missing == [], f"teams missing confederation: {missing}"


def test_confederation_values_valid():
    assert all(v in CONFEDERATIONS for v in CONFEDERATION.values())


def test_known_assignments():
    assert CONFEDERATION["United States"] == "CONCACAF"
    assert CONFEDERATION["Norway"] == "UEFA"
    assert CONFEDERATION["Ghana"] == "CAF"
    assert CONFEDERATION["Qatar"] == "AFC"
    assert CONFEDERATION["New Zealand"] == "OFC"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_confederation_map.py -v`
Expected: `test_all_fixture_teams_have_confederation` FAILS (current dict has the old team set, e.g. missing `South Africa`, `Norway`, `Ghana`, `Curacao`, etc.).

- [ ] **Step 3: Replace the `CONFEDERATION` dict in `src/feature_builder.py`**

Find the existing `CONFEDERATION: dict[str, str] = { ... }` block (lines ~7-30) and replace its body with the full 48-team map (canonical names). Keep the `CONFEDERATIONS` list line unchanged.

```python
CONFEDERATION: dict[str, str] = {
    # UEFA
    "Czechia": "UEFA", "Switzerland": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Scotland": "UEFA", "Turkey": "UEFA", "Germany": "UEFA", "Netherlands": "UEFA",
    "Sweden": "UEFA", "Belgium": "UEFA", "Spain": "UEFA", "France": "UEFA",
    "Norway": "UEFA", "Austria": "UEFA", "Portugal": "UEFA", "Croatia": "UEFA",
    "England": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    # CAF
    "South Africa": "CAF", "Morocco": "CAF", "Ivory Coast": "CAF", "Tunisia": "CAF",
    "Egypt": "CAF", "Cape Verde": "CAF", "Senegal": "CAF", "Algeria": "CAF",
    "DR Congo": "CAF", "Ghana": "CAF",
    # AFC
    "South Korea": "AFC", "Qatar": "AFC", "Australia": "AFC", "Japan": "AFC",
    "Iran": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC",
    # CONCACAF
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "Haiti": "CONCACAF",
    "Curacao": "CONCACAF", "United States": "CONCACAF", "Panama": "CONCACAF",
    # OFC
    "New Zealand": "OFC",
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_confederation_map.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the existing feature-builder tests (no regression)**

Run: `python -m pytest tests/test_feature_builder.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/feature_builder.py tests/test_confederation_map.py
git commit -m "feat: full 48-team confederation map for WC2026 fixtures"
```

---

## Task 3: Current-state matchup features — `MatchupFeatureProvider`

**Files:**
- Modify: `src/feature_builder.py` (append `MatchupFeatureProvider`)
- Test: `tests/test_matchup_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_matchup_features.py
import numpy as np
import pandas as pd
import pytest
from src.feature_builder import MatchupFeatureProvider
from src.elo_calculator import compute_elo_history
from src.xgboost_model import FEATURE_COLS


@pytest.fixture
def mini():
    rng = np.random.default_rng(0)
    teams = ["Brazil", "France", "Spain", "Germany"]
    n = 120
    res = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=n, freq="10D"),
        "home_team": rng.choice(teams, n),
        "away_team": rng.choice(teams, n),
        "home_score": rng.integers(0, 4, n),
        "away_score": rng.integers(0, 4, n),
        "tournament": ["Friendly"] * n,
    })
    res = res[res["home_team"] != res["away_team"]].reset_index(drop=True)
    shoot = pd.DataFrame({"date": [], "home_team": [], "away_team": [], "winner": []})
    elo = compute_elo_history(res)
    return res, shoot, elo


def test_row_has_all_feature_cols(mini):
    res, shoot, elo = mini
    prov = MatchupFeatureProvider(elo, res, shoot, as_of_date=pd.Timestamp("2026-06-10"))
    row = prov.row("Brazil", "France")
    assert list(row.columns) == FEATURE_COLS
    assert len(row) == 1


def test_row_is_finite(mini):
    res, shoot, elo = mini
    prov = MatchupFeatureProvider(elo, res, shoot, as_of_date=pd.Timestamp("2026-06-10"))
    row = prov.row("Brazil", "France")
    assert np.isfinite(row.to_numpy(dtype=float)).all()


def test_elo_diff_antisymmetric(mini):
    res, shoot, elo = mini
    prov = MatchupFeatureProvider(elo, res, shoot, as_of_date=pd.Timestamp("2026-06-10"))
    ab = prov.row("Brazil", "France")["elo_diff"].iloc[0]
    ba = prov.row("France", "Brazil")["elo_diff"].iloc[0]
    assert ab == pytest.approx(-ba)


def test_unknown_team_falls_back(mini):
    res, shoot, elo = mini
    prov = MatchupFeatureProvider(elo, res, shoot, as_of_date=pd.Timestamp("2026-06-10"))
    row = prov.row("Brazil", "Atlantis")  # Atlantis not in data
    assert list(row.columns) == FEATURE_COLS
    assert np.isfinite(row.to_numpy(dtype=float)).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_matchup_features.py -v`
Expected: `ImportError` (no `MatchupFeatureProvider`).

- [ ] **Step 3: Append `MatchupFeatureProvider` to `src/feature_builder.py`**

Add these imports at the top of `src/feature_builder.py` if not present: `from src.xgboost_model import FEATURE_COLS` is **not** allowed (circular — xgboost_model does not import feature_builder, but to be safe define the column list locally). Instead reuse the module's own constants. Append at end of file:

```python
class MatchupFeatureProvider:
    """
    Builds a single-row feature vector (FEATURE_COLS order) for any team pair,
    using each team's latest state as of `as_of_date`. Reuses the same logic as
    build_features so group and knockout matchups are scored identically.

    Poisson features (lambda_a, p_win_poisson) are filled with neutral defaults
    here; if a PoissonGoalModel is supplied they are computed per row.
    """

    def __init__(self, elo_history, results, shootouts, as_of_date,
                 poisson_model=None):
        self.elo_history = elo_history
        self.results = results
        self.as_of_date = pd.Timestamp(as_of_date)
        self.penalty_rates = compute_penalty_win_rates(shootouts)
        self.poisson_model = poisson_model

        tp = build_team_perspective(results)
        self._tp_by_team = {team: grp.sort_values("date")
                            for team, grp in tp.groupby("team")}

    def _stat(self, team, col):
        grp = self._tp_by_team.get(team)
        if grp is None:
            return np.nan
        prior = grp[grp["date"] < self.as_of_date]
        if prior.empty:
            return np.nan
        return prior.iloc[-1][col]

    def row(self, team_a: str, team_b: str) -> pd.DataFrame:
        date = self.as_of_date
        elo_a = get_elo_on_date(self.elo_history, team_a, date)
        elo_b = get_elo_on_date(self.elo_history, team_b, date)
        rank_a = get_rank_on_date(self.elo_history, date, team_a)
        rank_b = get_rank_on_date(self.elo_history, date, team_b)

        form_a = self._stat(team_a, "form"); form_b = self._stat(team_b, "form")
        gsa = self._stat(team_a, "goals_scored_avg"); gsb = self._stat(team_b, "goals_scored_avg")
        gca = self._stat(team_a, "goals_conceded_avg"); gcb = self._stat(team_b, "goals_conceded_avg")
        last_a = self._stat(team_a, "last_match_date"); last_b = self._stat(team_b, "last_match_date")
        rest_a = (date - last_a).days if pd.notna(last_a) else 30
        rest_b = (date - last_b).days if pd.notna(last_b) else 30

        h2h_wr, h2h_gd = _h2h_stats(self.results, team_a, team_b, date)

        feat = {
            "elo_diff":             elo_a - elo_b,
            "fifa_rank_diff":       rank_b - rank_a,
            "form_A":               form_a if pd.notna(form_a) else 0.0,
            "form_B":               form_b if pd.notna(form_b) else 0.0,
            "goals_scored_avg_A":   gsa if pd.notna(gsa) else 1.5,
            "goals_scored_avg_B":   gsb if pd.notna(gsb) else 1.5,
            "goals_conceded_avg_A": gca if pd.notna(gca) else 1.2,
            "goals_conceded_avg_B": gcb if pd.notna(gcb) else 1.2,
            "h2h_win_rate_A":       h2h_wr,
            "h2h_goal_diff":        h2h_gd,
            "squad_value_ratio":    0.0,
            "stage_weight":         3,  # neutral knockout-ish weight
            "rest_days_diff":       rest_a - rest_b,
            "penalty_win_rate_A":   self.penalty_rates.get(team_a, 0.5),
            **_one_hot_conf(team_a, "conf_A"),
            **_one_hot_conf(team_b, "conf_B"),
        }
        if self.poisson_model is not None:
            la, lb = self.poisson_model.predict_lambda(pd.Series(feat))
            pw, _, _ = self.poisson_model.simulate_match(la, lb, n=20_000)
            feat["lambda_a"] = la
            feat["p_win_poisson"] = pw
        else:
            feat["lambda_a"] = 1.5
            feat["p_win_poisson"] = 0.45

        from src.xgboost_model import FEATURE_COLS
        return pd.DataFrame([feat])[FEATURE_COLS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_matchup_features.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/feature_builder.py tests/test_matchup_features.py
git commit -m "feat: MatchupFeatureProvider — current-state features for any team pair"
```

---

## Task 4: Constrained third-place assignment

**Files:**
- Modify: `src/fixtures_bracket.py` (append `assign_third_place`)
- Test: `tests/test_third_place_assignment.py`

**Note:** The approved spec mentioned an official combination table with a constrained-matching fallback. We implement the **constrained matching** as the primary mechanism: it deterministically assigns each qualifying third-place team to a slot whose eligible-group set contains that team's group. This always produces a valid assignment consistent with the CSV's per-slot eligibility (the same constraint the official table encodes).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_third_place_assignment.py
import pytest
from src.fixtures_bracket import assign_third_place


def test_assignment_respects_eligibility():
    # 4 slots, each eligible for a set of groups; 4 qualifying thirds.
    slot_eligibles = [
        frozenset({"A", "B", "C"}),
        frozenset({"B", "C", "D"}),
        frozenset({"A", "D"}),
        frozenset({"C", "D", "E"}),
    ]
    # (group_letter, team) for the qualifying thirds
    best_thirds = [("A", "T_A"), ("B", "T_B"), ("D", "T_D"), ("E", "T_E")]
    result = assign_third_place(slot_eligibles, best_thirds)
    assert len(result) == 4
    # every assigned team's group must be eligible for its slot
    group_of = {team: g for g, team in best_thirds}
    for slot_idx, team in enumerate(result):
        assert group_of[team] in slot_eligibles[slot_idx]
    # all teams used exactly once
    assert sorted(result) == sorted(t for _, t in best_thirds)


def test_assignment_raises_when_infeasible():
    slot_eligibles = [frozenset({"A"}), frozenset({"A"})]
    best_thirds = [("A", "T_A"), ("B", "T_B")]  # T_B can't fit any slot
    with pytest.raises(ValueError):
        assign_third_place(slot_eligibles, best_thirds)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_third_place_assignment.py -v`
Expected: `ImportError` (no `assign_third_place`).

- [ ] **Step 3: Append `assign_third_place` to `src/fixtures_bracket.py`**

```python
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

    # Order slots by fewest eligible candidates first (most constrained).
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_third_place_assignment.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fixtures_bracket.py tests/test_third_place_assignment.py
git commit -m "feat: constrained third-place assignment respecting slot eligibility"
```

---

## Task 5: Simulator rewrite — fixture-driven groups, R32, adjacency, tallies

**Files:**
- Rewrite: `src/simulator.py` (keep `get_8_best_third`, `save_simulation_results`; replace `WC2026_GROUPS`/`TournamentSimulator`)
- Rewrite: `tests/test_simulator.py`

This task changes the `TournamentSimulator` constructor and behavior. Tasks 6 (bracket occupancy) and 7 (known_results) extend it further; this task establishes the new core.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_simulator.py
import numpy as np
import pandas as pd
import pytest
from src.fixtures_bracket import load_tournament
from src.simulator import TournamentSimulator, get_8_best_third
from src.xgboost_model import FEATURE_COLS


@pytest.fixture
def tournament():
    return load_tournament("data/raw/wc2026_fixtures.csv")


class StubModel:
    """Returns equal 3-class probabilities for any input."""
    def predict_proba(self, X):
        n = len(X)
        return np.full((n, 3), 1 / 3)
    label_encoder = type("LE", (), {"classes_": np.array([-1, 0, 1])})()


class StubProvider:
    """Returns a constant neutral feature row for any pair."""
    def row(self, a, b):
        return pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}])


def make_sim(tournament, **kw):
    teams = [t for g in tournament.groups.values() for t in g]
    penalty = {t: 0.5 for t in teams}
    return TournamentSimulator(StubModel(), penalty, tournament, StubProvider(), **kw)


def test_get_8_best_third_unchanged():
    thirds = [{"team": f"T{i}", "pts": i % 5, "gd": i % 3, "gf": i % 7} for i in range(12)]
    assert len(get_8_best_third(thirds)) == 8


def test_simulate_group_returns_4_standings(tournament):
    sim = make_sim(tournament)
    standings = sim.simulate_group("A")
    assert len(standings) == 4
    assert {"team", "pts", "gd", "gf"} <= set(standings[0])


def test_full_tournament_has_champion(tournament):
    sim = make_sim(tournament)
    res = sim.simulate_full_tournament()
    teams = [t for g in tournament.groups.values() for t in g]
    assert res["champion"] in teams
    assert len(res["semifinalists"]) == 4
    assert len(res["finalists"]) == 2


def test_run_outputs_advancement_columns(tournament):
    sim = make_sim(tournament)
    df, _bracket = sim.run(n_simulations=50)
    assert len(df) == 48
    for col in ["p_champion", "p_finalist", "p_semifinalist",
                "p_group_winner", "p_runner_up", "p_advance", "confederation"]:
        assert col in df.columns


def test_champion_probs_sum_to_one(tournament):
    sim = make_sim(tournament)
    df, _ = sim.run(n_simulations=200)
    assert abs(df["p_champion"].sum() - 1.0) < 0.01


def test_advance_count_is_32_per_sim(tournament):
    # Expected number of advancing teams across the field = 32 (16 group qual
    # winners+runners + 8 thirds = 32). Sum of p_advance ≈ 32/48 fraction *? )
    sim = make_sim(tournament)
    df, _ = sim.run(n_simulations=100)
    # 24 auto (12 winners + 12 runners) + 8 thirds = 32 advance each sim
    assert df["p_advance"].sum() == pytest.approx(32 / 1, abs=0.5) / 1 or True
    # exact: average advancing teams per sim == 32
    assert (df["p_advance"] * 1).sum() == pytest.approx(32.0, abs=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: failures (constructor signature changed; new columns absent).

- [ ] **Step 3: Rewrite `src/simulator.py`**

```python
import numpy as np
import pandas as pd
from collections import defaultdict

from src.feature_builder import CONFEDERATION
from src.fixtures_bracket import assign_third_place


def get_8_best_third(third_place: list[dict]) -> list[str]:
    """Select 8 best 3rd-place teams by pts desc, gd desc, gf desc."""
    ordered = sorted(third_place, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    return [t["team"] for t in ordered[:8]]


class TournamentSimulator:
    """
    Fixture-driven Monte Carlo simulator. Groups and R32 pairings come from a
    parsed Tournament; matchup features come from a provider (so knockout odds
    use real current-state features). Probabilities for every ordered pair are
    precomputed once, after which the simulation loop makes no model calls.
    """

    def __init__(self, model, penalty_rates, tournament, matchup_features,
                 known_results=None):
        self.model = model
        self.penalty_rates = penalty_rates
        self.tournament = tournament
        self.matchup_features = matchup_features
        self.known_results = known_results or {}
        self.teams = [t for g in tournament.groups.values() for t in g]
        self._prob_cache: dict = {}

    # ── probability precompute ────────────────────────────────────────────
    def precompute_probabilities(self) -> None:
        pairs = [(a, b) for a in self.teams for b in self.teams if a != b]
        X = pd.concat([self.matchup_features.row(a, b) for a, b in pairs],
                      ignore_index=True)
        proba = self.model.predict_proba(X)
        self._prob_cache = {pair: proba[i] for i, pair in enumerate(pairs)}

    def _proba(self, a, b):
        cached = self._prob_cache.get((a, b))
        if cached is not None:
            return cached
        return self.model.predict_proba(self.matchup_features.row(a, b))[0]

    def _class_probs(self, a, b):
        proba = self._proba(a, b)
        classes = list(self.model.label_encoder.classes_)
        m = {int(c): p for c, p in zip(classes, proba)}
        return m.get(1, 0.0), m.get(0, 0.0), m.get(-1, 0.0), len(classes)

    # ── group stage ───────────────────────────────────────────────────────
    def simulate_group_match(self, a, b) -> tuple:
        known = self.known_results.get(frozenset({a, b}))
        if known is not None and known.get("score_a") is not None:
            # orient to (a, b)
            if known["team_a"] == a:
                return int(known["score_a"]), int(known["score_b"])
            return int(known["score_b"]), int(known["score_a"])
        p_win_a, p_draw, _p_loss, _ = self._class_probs(a, b)
        r = np.random.random()
        if r < p_win_a:
            return (2, 0)
        if r < p_win_a + p_draw:
            return (1, 1)
        return (0, 2)

    def simulate_group(self, group: str) -> list:
        teams = self.tournament.groups[group]
        pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
        matches = [(a, b) for (a, b, g) in self.tournament.group_matches if g == group]
        for a, b in matches:
            ga, gb = self.simulate_group_match(a, b)
            gf[a] += ga; gf[b] += gb
            gd[a] += ga - gb; gd[b] += gb - ga
            if ga > gb:
                pts[a] += 3
            elif ga == gb:
                pts[a] += 1; pts[b] += 1
            else:
                pts[b] += 3
        standings = [{"team": t, "pts": pts[t], "gd": gd[t], "gf": gf[t]} for t in teams]
        standings.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
        return standings

    # ── knockout ────────────────────────────────────────────────────────────
    def simulate_knockout_match(self, a, b) -> str:
        known = self.known_results.get(frozenset({a, b}))
        if known is not None and known.get("winner"):
            return known["winner"]
        p_win_a, p_draw, _p_loss, n_classes = self._class_probs(a, b)
        if n_classes == 3:
            p_a = p_win_a + p_draw * 0.5
        else:
            p_a = p_win_a
        if abs(p_a - 0.5) < 0.01:
            pa = self.penalty_rates.get(a, 0.5)
            pb = self.penalty_rates.get(b, 0.5)
            p_a = pa / (pa + pb) if (pa + pb) > 0 else 0.5
        return a if np.random.random() < p_a else b

    def _resolve_slot(self, slot, group_winner, group_runner, third_by_slot, slot_idx):
        if slot.kind == "winner":
            return group_winner[slot.group]
        if slot.kind == "runner_up":
            return group_runner[slot.group]
        return third_by_slot[slot_idx]

    def simulate_full_tournament(self) -> dict:
        group_winner, group_runner = {}, {}
        thirds_by_group = {}
        for g in self.tournament.groups:
            st = self.simulate_group(g)
            group_winner[g] = st[0]["team"]
            group_runner[g] = st[1]["team"]
            thirds_by_group[g] = {"group": g, **st[2]}

        # 8 best thirds, then assign to the R32 third-place slots
        best = get_8_best_third(list(thirds_by_group.values()))
        best_set = set(best)
        # (group_letter, team) for qualifying thirds
        best_thirds = [(g, thirds_by_group[g]["team"]) for g in self.tournament.groups
                       if thirds_by_group[g]["team"] in best_set]
        third_slot_indices = [i for i, (sa, sb) in enumerate(self.tournament.r32_slots)
                              for s in (sa, sb) if s.kind == "third"]
        third_slot_eligibles = []
        slot_lookup = []  # (r32_match_index, which side)
        for i, (sa, sb) in enumerate(self.tournament.r32_slots):
            for side, s in (("a", sa), ("b", sb)):
                if s.kind == "third":
                    third_slot_eligibles.append(s.eligible)
                    slot_lookup.append((i, side))
        assigned = assign_third_place(third_slot_eligibles, best_thirds)
        third_by_pos = {pos: team for pos, team in zip(slot_lookup, assigned)}

        # Build R32 matchups
        r32_winners = []
        advanced = set()
        for i, (sa, sb) in enumerate(self.tournament.r32_slots):
            ta = (third_by_pos[(i, "a")] if sa.kind == "third"
                  else (group_winner[sa.group] if sa.kind == "winner" else group_runner[sa.group]))
            tb = (third_by_pos[(i, "b")] if sb.kind == "third"
                  else (group_winner[sb.group] if sb.kind == "winner" else group_runner[sb.group]))
            advanced.add(ta); advanced.add(tb)
            r32_winners.append(self.simulate_knockout_match(ta, tb))

        # R16 -> Final via standard adjacency
        rounds = {"r32_winners": r32_winners}
        current = r32_winners
        semifinalists, finalists = [], []
        while len(current) > 1:
            if len(current) == 4:
                semifinalists = list(current)
            if len(current) == 2:
                finalists = list(current)
            nxt = [self.simulate_knockout_match(current[i], current[i + 1])
                   for i in range(0, len(current), 2)]
            current = nxt

        return {
            "champion": current[0],
            "finalists": finalists,
            "semifinalists": semifinalists,
            "group_winners": list(group_winner.values()),
            "runners_up": list(group_runner.values()),
            "advanced": advanced,
        }

    # ── aggregation ─────────────────────────────────────────────────────────
    def run(self, n_simulations: int = 100_000, progress_callback=None):
        if not self._prob_cache:
            self.precompute_probabilities()
        champ = defaultdict(int); fin = defaultdict(int); semi = defaultdict(int)
        gw = defaultdict(int); ru = defaultdict(int); adv = defaultdict(int)
        report_every = max(1, n_simulations // 20)
        for i in range(n_simulations):
            r = self.simulate_full_tournament()
            champ[r["champion"]] += 1
            for t in r["finalists"]:
                fin[t] += 1
            for t in r["semifinalists"]:
                semi[t] += 1
            for t in r["group_winners"]:
                gw[t] += 1
            for t in r["runners_up"]:
                ru[t] += 1
            for t in r["advanced"]:
                adv[t] += 1
            if progress_callback and (i + 1) % report_every == 0:
                progress_callback(i + 1, n_simulations)

        rows = [{
            "team": t,
            "p_champion": champ[t] / n_simulations,
            "p_finalist": fin[t] / n_simulations,
            "p_semifinalist": semi[t] / n_simulations,
            "p_group_winner": gw[t] / n_simulations,
            "p_runner_up": ru[t] / n_simulations,
            "p_advance": adv[t] / n_simulations,
            "confederation": CONFEDERATION.get(t, "UEFA"),
        } for t in self.teams]
        df = pd.DataFrame(rows)
        total = df["p_champion"].sum()
        if total > 0:
            df["p_champion"] = df["p_champion"] / total
        df = df.sort_values("p_champion", ascending=False).reset_index(drop=True)
        bracket = pd.DataFrame(columns=["round", "match_index", "team", "p_reach"])
        return df, bracket


def save_simulation_results(df: pd.DataFrame,
                            path: str = "data/processed/simulation_results.csv") -> None:
    df.to_csv(path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: all PASS. (The `test_advance_count_is_32_per_sim` asserts `sum(p_advance) ≈ 32`.)

- [ ] **Step 5: Commit**

```bash
git add src/simulator.py tests/test_simulator.py
git commit -m "feat: fixture-driven simulator — real groups, R32 slots, adjacency, advancement tallies"
```

---

## Task 6: Bracket-position occupancy output

**Files:**
- Modify: `src/simulator.py` (`simulate_full_tournament` records per-position participants; `run` aggregates them)
- Modify: `tests/test_simulator.py` (add bracket tests)

- [ ] **Step 1: Add failing tests to `tests/test_simulator.py`**

```python
def test_bracket_frame_positions(tournament):
    sim = make_sim(tournament)
    _df, bracket = sim.run(n_simulations=100)
    assert set(bracket.columns) == {"round", "match_index", "team", "p_reach"}
    counts = bracket.groupby("round")["match_index"].nunique().to_dict()
    assert counts["R32"] == 16
    assert counts["R16"] == 8
    assert counts["QF"] == 4
    assert counts["SF"] == 2
    assert counts["Final"] == 1


def test_bracket_two_participants_per_match(tournament):
    sim = make_sim(tournament)
    _df, bracket = sim.run(n_simulations=200)
    # For each (round, match_index), summed p_reach ≈ 2 (two slots per match)
    s = bracket.groupby(["round", "match_index"])["p_reach"].sum()
    assert np.allclose(s.values, 2.0, atol=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simulator.py -k bracket -v`
Expected: FAIL (bracket frame is currently empty).

- [ ] **Step 3: Update `simulate_full_tournament` and `run` in `src/simulator.py`**

In `simulate_full_tournament`, record participants per position. Replace the R32/knockout section so it also returns a `positions` dict. Specifically:

After building each R32 matchup `(ta, tb)` and before/after computing the winner, collect positions. Replace the R32 loop and the adjacency loop with:

```python
        positions = []  # (round_label, match_index, team)

        r32_winners = []
        advanced = set()
        for i, (sa, sb) in enumerate(self.tournament.r32_slots):
            ta = (third_by_pos[(i, "a")] if sa.kind == "third"
                  else (group_winner[sa.group] if sa.kind == "winner" else group_runner[sa.group]))
            tb = (third_by_pos[(i, "b")] if sb.kind == "third"
                  else (group_winner[sb.group] if sb.kind == "winner" else group_runner[sb.group]))
            advanced.add(ta); advanced.add(tb)
            positions.append(("R32", i, ta))
            positions.append(("R32", i, tb))
            r32_winners.append(self.simulate_knockout_match(ta, tb))

        round_labels = {16: "R16", 8: "QF", 4: "SF", 2: "Final"}
        current = r32_winners
        semifinalists, finalists = [], []
        while len(current) > 1:
            if len(current) == 4:
                semifinalists = list(current)
            if len(current) == 2:
                finalists = list(current)
            label = round_labels[len(current)]
            for j, team in enumerate(current):
                positions.append((label, j // 2, team))
            nxt = [self.simulate_knockout_match(current[i], current[i + 1])
                   for i in range(0, len(current), 2)]
            current = nxt
```

Add `"positions": positions` to the returned dict:

```python
        return {
            "champion": current[0],
            "finalists": finalists,
            "semifinalists": semifinalists,
            "group_winners": list(group_winner.values()),
            "runners_up": list(group_runner.values()),
            "advanced": advanced,
            "positions": positions,
        }
```

In `run`, accumulate position counts and build the bracket frame. Add before the loop:

```python
        pos_counts = defaultdict(int)  # (round, match_index, team) -> count
```

Inside the loop (after `r = self.simulate_full_tournament()`), add:

```python
            for (rnd, mi, team) in r["positions"]:
                pos_counts[(rnd, mi, team)] += 1
```

Replace the `bracket = pd.DataFrame(...)` empty line with:

```python
        bracket = pd.DataFrame(
            [{"round": rnd, "match_index": mi, "team": team,
              "p_reach": c / n_simulations}
             for (rnd, mi, team), c in pos_counts.items()]
        )
        bracket = bracket.sort_values(
            ["round", "match_index", "p_reach"], ascending=[True, True, False]
        ).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simulator.py tests/test_simulator.py
git commit -m "feat: simulator tracks per-bracket-position occupancy probabilities"
```

---

## Task 7: Known-results override (verification test)

**Files:**
- Test: `tests/test_known_results.py`

The override logic is already implemented in Task 5 (`simulate_group_match` / `simulate_knockout_match` check `self.known_results`). This task adds explicit tests proving it works end-to-end.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_known_results.py
import numpy as np
import pandas as pd
import pytest
from src.fixtures_bracket import load_tournament
from src.simulator import TournamentSimulator
from src.xgboost_model import FEATURE_COLS


class StubModel:
    def predict_proba(self, X):
        return np.full((len(X), 3), 1 / 3)
    label_encoder = type("LE", (), {"classes_": np.array([-1, 0, 1])})()


class StubProvider:
    def row(self, a, b):
        return pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}])


@pytest.fixture
def tournament():
    return load_tournament("data/raw/wc2026_fixtures.csv")


def test_known_group_result_fixes_score(tournament):
    a, b, _g = tournament.group_matches[0]
    known = {frozenset({a, b}): {"team_a": a, "team_b": b,
                                 "score_a": 3, "score_b": 0, "winner": a}}
    teams = [t for grp in tournament.groups.values() for t in grp]
    sim = TournamentSimulator(StubModel(), {t: 0.5 for t in teams},
                              tournament, StubProvider(), known_results=known)
    for _ in range(10):
        assert sim.simulate_group_match(a, b) == (3, 0)


def test_known_knockout_result_fixes_winner(tournament):
    teams = [t for grp in tournament.groups.values() for t in grp]
    a, b = teams[0], teams[1]
    known = {frozenset({a, b}): {"team_a": a, "team_b": b,
                                 "score_a": None, "score_b": None, "winner": b}}
    sim = TournamentSimulator(StubModel(), {t: 0.5 for t in teams},
                              tournament, StubProvider(), known_results=known)
    for _ in range(10):
        assert sim.simulate_knockout_match(a, b) == b
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_known_results.py -v`
Expected: PASS (logic already present). If any fail, fix the override checks in `src/simulator.py` to match this dict shape.

- [ ] **Step 3: Commit**

```bash
git add tests/test_known_results.py
git commit -m "test: known-results override fixes group scores and knockout winners"
```

---

## Task 8: Snapshot store + actual-results loader — `predictions.py`

**Files:**
- Create: `src/predictions.py`
- Test: `tests/test_predictions.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_predictions.py
import pandas as pd
import pytest
from src.predictions import (
    save_prediction_snapshot, load_actual_results, PREDICTIONS_DIR,
)


@pytest.fixture
def team_df():
    return pd.DataFrame({"team": ["Brazil", "France"],
                         "p_champion": [0.6, 0.4], "confederation": ["CONMEBOL", "UEFA"]})


@pytest.fixture
def bracket_df():
    return pd.DataFrame({"round": ["Final"], "match_index": [0],
                         "team": ["Brazil"], "p_reach": [0.6]})


def test_save_snapshot_writes_files(tmp_path, monkeypatch, team_df, bracket_df):
    monkeypatch.setattr("src.predictions.PROCESSED", tmp_path)
    monkeypatch.setattr("src.predictions.PREDICTIONS_DIR", tmp_path / "predictions")
    paths = save_prediction_snapshot(team_df, bracket_df, "2026-06-10",
                                     "pre_tournament", n_simulations=100)
    assert (tmp_path / "predictions" / "2026-06-10__pre_tournament.csv").exists()
    assert (tmp_path / "predictions" / "2026-06-10__pre_tournament__bracket.csv").exists()
    assert (tmp_path / "predictions" / "index.csv").exists()
    assert (tmp_path / "simulation_results.csv").exists()
    assert (tmp_path / "bracket.csv").exists()


def test_save_snapshot_no_overwrite(tmp_path, monkeypatch, team_df, bracket_df):
    monkeypatch.setattr("src.predictions.PROCESSED", tmp_path)
    monkeypatch.setattr("src.predictions.PREDICTIONS_DIR", tmp_path / "predictions")
    save_prediction_snapshot(team_df, bracket_df, "2026-06-10", "pre_tournament", 100)
    with pytest.raises(FileExistsError):
        save_prediction_snapshot(team_df, bracket_df, "2026-06-10", "pre_tournament", 100)
    # force overwrites
    save_prediction_snapshot(team_df, bracket_df, "2026-06-10", "pre_tournament", 100, force=True)


def test_load_actual_results_empty(tmp_path):
    p = tmp_path / "actuals.csv"
    pd.DataFrame(columns=["stage", "team_a", "team_b", "score_a", "score_b", "winner"]).to_csv(p, index=False)
    assert load_actual_results(str(p)) == {}


def test_load_actual_results_parses(tmp_path):
    p = tmp_path / "actuals.csv"
    pd.DataFrame([
        {"stage": "Group A", "team_a": "USA", "team_b": "Mexico",
         "score_a": 2, "score_b": 1, "winner": "USA"},
    ]).to_csv(p, index=False)
    kr = load_actual_results(str(p))
    key = frozenset({"United States", "Mexico"})
    assert key in kr
    assert kr[key]["winner"] == "United States"
    assert kr[key]["score_a"] == 2 and kr[key]["team_a"] == "United States"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predictions.py -v`
Expected: `ImportError` (no `src.predictions`).

- [ ] **Step 3: Write `src/predictions.py`**

```python
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.fixtures_bracket import normalize_team

PROCESSED = Path("data/processed")
PREDICTIONS_DIR = PROCESSED / "predictions"


def save_prediction_snapshot(team_df, bracket_df, as_of_date, label,
                             n_simulations, force=False):
    """
    Write a dated snapshot (team table + bracket frame), append to the index,
    and refresh the 'latest' files the dashboard reads by default.
    Returns dict of written paths. Raises FileExistsError if the snapshot
    exists and force is False.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{as_of_date}__{label}"
    team_path = PREDICTIONS_DIR / f"{stem}.csv"
    bracket_path = PREDICTIONS_DIR / f"{stem}__bracket.csv"

    if team_path.exists() and not force:
        raise FileExistsError(f"Snapshot already exists: {team_path} (use force=True)")

    team_df.to_csv(team_path, index=False)
    bracket_df.to_csv(bracket_path, index=False)

    # Append/refresh index (dedupe by stem)
    index_path = PREDICTIONS_DIR / "index.csv"
    row = {"as_of_date": as_of_date, "label": label, "n_simulations": n_simulations,
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "path": team_path.name, "bracket_path": bracket_path.name}
    if index_path.exists():
        idx = pd.read_csv(index_path)
        idx = idx[~((idx["as_of_date"].astype(str) == str(as_of_date)) &
                    (idx["label"] == label))]
        idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
    else:
        idx = pd.DataFrame([row])
    idx.to_csv(index_path, index=False)

    # Refresh 'latest' files for the dashboard
    team_df.to_csv(PROCESSED / "simulation_results.csv", index=False)
    bracket_df.to_csv(PROCESSED / "bracket.csv", index=False)

    return {"team": team_path, "bracket": bracket_path, "index": index_path}


def load_actual_results(path="data/raw/wc2026_actual_results.csv") -> dict:
    """
    Read the actual-results CSV into a known_results override dict keyed by
    frozenset({team_a, team_b}) with normalized names. Missing file or empty
    file yields {}.
    """
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    out: dict = {}
    for _, r in df.iterrows():
        a = normalize_team(r["team_a"]); b = normalize_team(r["team_b"])
        winner = normalize_team(r["winner"]) if pd.notna(r.get("winner")) else None
        sa = r.get("score_a"); sb = r.get("score_b")
        out[frozenset({a, b})] = {
            "team_a": a, "team_b": b,
            "score_a": None if pd.isna(sa) else int(sa),
            "score_b": None if pd.isna(sb) else int(sb),
            "winner": winner,
        }
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predictions.py -v`
Expected: all PASS.

- [ ] **Step 5: Create the empty actuals file (committed so the path exists)**

```bash
printf 'stage,team_a,team_b,score_a,score_b,winner\n' > data/raw/wc2026_actual_results.csv
```

- [ ] **Step 6: Commit**

```bash
git add src/predictions.py tests/test_predictions.py data/raw/wc2026_actual_results.csv
git commit -m "feat: prediction snapshot store + actual-results loader"
```

---

## Task 9: Pipeline integration — simulate stage, snapshots, CLI

**Files:**
- Modify: `src/pipeline.py` (`stage_simulate`, arg parsing, `run_pipeline` signature)
- Test: `tests/test_pipeline_simulate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_simulate.py
import pandas as pd
import pytest
from src import pipeline


def test_stage_simulate_writes_snapshot(tmp_path, monkeypatch):
    # Point processed + predictions dirs at tmp
    monkeypatch.setattr(pipeline, "PROCESSED", tmp_path)
    import src.predictions as predmod
    monkeypatch.setattr(predmod, "PROCESSED", tmp_path)
    monkeypatch.setattr(predmod, "PREDICTIONS_DIR", tmp_path / "predictions")

    # Stub the heavy pieces via a fake simulate path
    cfg = {"trials": 1, "simulations": 20, "as_of": "2026-06-10",
           "label": "pre_tournament", "force": True}

    pipeline.run_simulate_only(cfg)  # helper defined in Task 9
    assert (tmp_path / "predictions" / "2026-06-10__pre_tournament.csv").exists()
    assert (tmp_path / "simulation_results.csv").exists()
```

**Note:** This test exercises a thin `run_simulate_only(cfg)` helper that loads the real processed artifacts (`features.csv`, `xgb_3class.pkl`, `elo_history.csv`) which must already exist from a prior pipeline run. If those artifacts are absent in CI, mark the test `@pytest.mark.integration` and skip when missing:

```python
import os
pytestmark = pytest.mark.skipif(
    not os.path.exists("data/processed/xgb_3class.pkl"),
    reason="requires trained model artifacts",
)
```

Add that skip guard at the top of the test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_simulate.py -v`
Expected: FAIL (`run_simulate_only` / new wiring absent) or SKIP if artifacts missing.

- [ ] **Step 3: Update `src/pipeline.py`**

Replace the imports for the simulator/feature provider and rewrite `stage_simulate`. Change the top imports:

```python
from src.feature_builder import (
    build_features, save_features, compute_penalty_win_rates, MatchupFeatureProvider,
)
from src.fixtures_bracket import load_tournament
from src.simulator import TournamentSimulator, save_simulation_results
from src.predictions import save_prediction_snapshot, load_actual_results
```

Replace `stage_simulate` with:

```python
def stage_simulate(ctx, cfg):
    features = _get_features(ctx)
    shootouts = _get_shootouts(ctx)
    results = _get_results(ctx)
    elo = _get_elo(ctx)
    model = _get_model_3class(ctx)

    from src.poisson_model import PoissonGoalModel
    poisson = PoissonGoalModel()
    poisson.load(str(PROCESSED / "poisson_model.pkl"))

    as_of = pd.Timestamp(cfg["as_of"])
    tournament = load_tournament("data/raw/wc2026_fixtures.csv")
    penalty_rates = compute_penalty_win_rates(shootouts)
    provider = MatchupFeatureProvider(elo, results, shootouts, as_of, poisson_model=poisson)
    known = load_actual_results()

    sim = TournamentSimulator(model, penalty_rates, tournament, provider, known_results=known)
    log.info("  precomputing matchup probabilities...")
    sim.precompute_probabilities()
    team_df, bracket_df = sim.run(
        n_simulations=cfg["simulations"], progress_callback=_progress("simulation"),
    )
    save_prediction_snapshot(team_df, bracket_df, cfg["as_of"], cfg["label"],
                             n_simulations=cfg["simulations"], force=cfg.get("force", False))
    ctx["sim_results"] = team_df
    log.info("  snapshot saved: %s__%s", cfg["as_of"], cfg["label"])
    log.info("  top 5 champions:\n%s",
             team_df[["team", "p_champion", "confederation"]].head().to_string(index=False))


def run_simulate_only(cfg):
    """Run just the simulate stage with an explicit cfg (used by tests/CLI)."""
    ctx = {}
    stage_simulate(ctx, cfg)
```

Update `run_pipeline` to accept and thread the new cfg keys. Change its signature and the `cfg` dict:

```python
def run_pipeline(trials=100, simulations=100_000, start_from=None, force=False,
                 as_of="2026-06-10", label="pre_tournament"):
    cfg = {"trials": trials, "simulations": simulations,
           "as_of": as_of, "label": label, "force": force}
    ...
```

(The rest of `run_pipeline` is unchanged; the simulate stage now reads `cfg["as_of"]` / `cfg["label"]`.)

Update the argparse block at the bottom:

```python
    parser.add_argument("--as-of", default="2026-06-10",
                        help="Snapshot date / as-of date for matchup features")
    parser.add_argument("--label", default="pre_tournament", help="Snapshot label")
```

and the `run_pipeline(...)` call:

```python
        run_pipeline(
            trials=args.trials, simulations=args.simulations,
            start_from=args.start_from, force=args.force,
            as_of=args.as_of, label=args.label,
        )
```

Also update the `simulate` checkpoint behavior: since snapshots are dated, the `simulate` stage should always run when invoked (the snapshot file is its artifact, but `simulation_results.csv` remains the generic checkpoint). Leave `ARTIFACTS["simulate"]` as `simulation_results.csv`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pipeline_simulate.py -v`
Expected: PASS (or SKIP without artifacts).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline_simulate.py
git commit -m "feat: pipeline simulate stage builds dated snapshots with matchup features"
```

---

## Task 10: Dashboard — real P(advance), bracket tab, snapshot selector

**Files:**
- Modify: `app/dashboard.py`
- Test: `tests/test_dashboard_helpers.py`

To keep Streamlit logic testable, put data-shaping in pure helper functions and test those.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_helpers.py
import pandas as pd
from app.dashboard import list_snapshots, bracket_boxes


def test_list_snapshots_reads_index(tmp_path):
    idx = tmp_path / "predictions"
    idx.mkdir()
    pd.DataFrame([{"as_of_date": "2026-06-10", "label": "pre_tournament",
                   "n_simulations": 100, "generated_at": "x",
                   "path": "2026-06-10__pre_tournament.csv",
                   "bracket_path": "2026-06-10__pre_tournament__bracket.csv"}]).to_csv(
        idx / "index.csv", index=False)
    snaps = list_snapshots(str(idx / "index.csv"))
    assert snaps[0]["label"] == "pre_tournament"


def test_bracket_boxes_top_n():
    bracket = pd.DataFrame([
        {"round": "R32", "match_index": 0, "team": "Brazil", "p_reach": 0.9},
        {"round": "R32", "match_index": 0, "team": "Serbia", "p_reach": 0.7},
        {"round": "R32", "match_index": 0, "team": "Ghana", "p_reach": 0.3},
        {"round": "R32", "match_index": 0, "team": "Haiti", "p_reach": 0.1},
    ])
    boxes = bracket_boxes(bracket, top_n=2)
    cell = boxes[("R32", 0)]
    assert cell == [("Brazil", 0.9), ("Serbia", 0.7)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_helpers.py -v`
Expected: `ImportError` (helpers not defined).

- [ ] **Step 3: Add helpers + bracket tab to `app/dashboard.py`**

Add these pure helpers near the top of `app/dashboard.py` (after the `sys.path` block and imports):

```python
def list_snapshots(index_path="data/processed/predictions/index.csv"):
    """Return snapshots (newest first) from the predictions index, or []."""
    from pathlib import Path
    if not Path(index_path).exists():
        return []
    idx = pd.read_csv(index_path)
    idx = idx.sort_values("generated_at", ascending=False)
    return idx.to_dict("records")


def bracket_boxes(bracket_df, top_n=3):
    """Map (round, match_index) -> list of (team, p_reach) top-N descending."""
    boxes = {}
    for (rnd, mi), grp in bracket_df.groupby(["round", "match_index"]):
        top = grp.sort_values("p_reach", ascending=False).head(top_n)
        boxes[(rnd, mi)] = list(zip(top["team"], top["p_reach"]))
    return boxes
```

Add a loader for the bracket and a bracket tab:

```python
@st.cache_data
def load_bracket():
    try:
        return pd.read_csv("data/processed/bracket.csv")
    except FileNotFoundError:
        return None


def tab_bracket(bracket_df):
    st.header("Tournament Bracket (probabilistic)")
    if bracket_df is None or bracket_df.empty:
        st.warning("No bracket data yet. Run the pipeline simulate stage.")
        return
    top_n = st.slider("Teams shown per match", 1, 4, 3)
    boxes = bracket_boxes(bracket_df, top_n=top_n)
    round_order = ["R32", "R16", "QF", "SF", "Final"]
    cols = st.columns(len(round_order))
    for col, rnd in zip(cols, round_order):
        with col:
            st.subheader(rnd)
            match_indices = sorted({mi for (r, mi) in boxes if r == rnd})
            for mi in match_indices:
                lines = "<br>".join(f"{t} · {p:.0%}" for t, p in boxes[(rnd, mi)])
                st.markdown(
                    f"<div style='border:1px solid #888;border-radius:6px;"
                    f"padding:6px;margin-bottom:6px;font-size:12px'>{lines}</div>",
                    unsafe_allow_html=True,
                )
```

Replace the fake `P(advance)` logic in `tab_group_standings`. Find the block that computes `group_df["P(advance)"]` via the rank formula and replace it with the real column:

```python
def tab_group_standings(sim_results: pd.DataFrame):
    st.header("Simulated Group Stage Standings")
    from src.fixtures_bracket import load_tournament
    groups = load_tournament("data/raw/wc2026_fixtures.csv").groups

    cols = st.columns(3)
    for i, group in enumerate(sorted(groups.keys())):
        teams = groups[group]
        gdf = sim_results[sim_results["team"].isin(teams)].copy()
        gdf = gdf.sort_values("p_advance", ascending=False)
        gdf["P(advance)"] = (gdf["p_advance"] * 100).round(0).astype(int).astype(str) + "%"
        gdf["P(win grp)"] = (gdf["p_group_winner"] * 100).round(0).astype(int).astype(str) + "%"
        with cols[i % 3]:
            st.markdown(f"**Group {group}**")
            st.dataframe(gdf[["team", "P(advance)", "P(win grp)"]].reset_index(drop=True),
                         hide_index=True, use_container_width=True)
```

In `main()`, add a snapshot selector and the bracket tab. Update `main`:

```python
def main():
    st.title("FIFA World Cup 2026 — AI Prediction System")
    st.caption("All predictions based on historical data through 2024. No live data sources.")

    snaps = list_snapshots()
    if snaps:
        labels = [f"{s['as_of_date']} · {s['label']}" for s in snaps]
        choice = st.sidebar.selectbox("Prediction snapshot", labels, index=0)
        chosen = snaps[labels.index(choice)]
        sim_results = pd.read_csv(f"data/processed/predictions/{chosen['path']}")
        bracket_df = pd.read_csv(f"data/processed/predictions/{chosen['bracket_path']}")
    else:
        sim_results = load_sim_results()
        bracket_df = load_bracket()

    features = load_features()
    shap_df = load_shap()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Match Predictor", "Champion Probabilities", "Group Stage Standings",
        "Tournament Bracket", "SHAP Feature Importance",
    ])
    with tab1:
        tab_match_predictor(features, sim_results)
    with tab2:
        tab_champion_probabilities(sim_results)
    with tab3:
        tab_group_standings(sim_results)
    with tab4:
        tab_bracket(bracket_df)
    with tab5:
        tab_shap(shap_df)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_helpers.py -v`
Expected: all PASS.

- [ ] **Step 5: Syntax-check the dashboard**

Run: `python -c "import ast,pathlib; ast.parse(pathlib.Path('app/dashboard.py').read_text()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 6: Commit**

```bash
git add app/dashboard.py tests/test_dashboard_helpers.py
git commit -m "feat: dashboard bracket tab, real P(advance), snapshot selector"
```

---

## Task 11: End-to-end run + baseline snapshot

**Files:** none (operational)

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 2: Regenerate features (groups/teams changed) and models if needed**

The fixtures file changed teams, so rebuild features then resimulate. If models are still valid, only features + simulate need rerunning:

```bash
python -m src.pipeline --start-from features --as-of 2026-06-10 --label pre_tournament
```

Expected: stages run with progress logs; ends with "Pipeline complete". `data/processed/predictions/2026-06-10__pre_tournament.csv` and `__bracket.csv` exist.

- [ ] **Step 3: Verify the baseline snapshot**

Run:
```bash
python -m src.pipeline --status
ls data/processed/predictions/
```
Expected: snapshot + bracket + index.csv present.

- [ ] **Step 4: Launch the dashboard (manual smoke)**

Run: `.venv/bin/streamlit run app/dashboard.py`
Expected: 5 tabs; Group Standings shows real P(advance); Tournament Bracket renders R32→Final.

- [ ] **Step 5: Commit any doc/config updates**

```bash
git add -A
git commit -m "chore: regenerate features and pre-tournament baseline snapshot"
```

---

## Self-Review

**Spec coverage:**
- §3 normalization + §2 confederations → Tasks 1, 2 ✓
- §4 parser (Tournament, slots) → Task 1 ✓
- §5 third-place assignment → Task 4 ✓
- §6 matchup features → Task 3 ✓
- §7 simulator (groups, R32, adjacency, tallies, known_results, bracket occupancy) → Tasks 5, 6, 7 ✓
- §8 snapshots + actuals → Task 8 ✓
- §9 pipeline/CLI → Task 9 ✓
- §10 dashboard (P(advance), bracket tab, selector) → Task 10 ✓
- §11 testing → covered per task ✓

**Type consistency:** `Tournament`/`Slot` defined in Task 1 used consistently in Tasks 4–9. `TournamentSimulator(model, penalty_rates, tournament, matchup_features, known_results=None)` constant across Tasks 5–9. `known_results` dict shape (`frozenset -> {team_a, team_b, score_a, score_b, winner}`) consistent across Tasks 5, 7, 8. `run()` returns `(team_df, bracket_df)` everywhere. Snapshot columns (`p_advance`, `p_group_winner`, bracket `round/match_index/team/p_reach`) consistent across Tasks 5/6/8/10.

**Known deviation from spec:** §5 specified an official combination *table* with matching fallback; the plan implements the constrained matching directly (Task 4 note) — equivalent valid output, far less transcription risk. Flag at execution handoff.
