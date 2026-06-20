# Fixture-Driven Tournament Simulation — Design Spec

**Date:** 2026-06-20
**Status:** Approved (pending spec review)
**Scope:** Make the Monte Carlo simulator a real WC 2026 prediction driven by `wc2026_fixtures.csv`, with dated prediction snapshots, round-by-round updates from real results, and a probabilistic tournament-bracket view.

---

## 1. Objective

Today the simulator uses a hardcoded `WC2026_GROUPS` dict, a generated round-robin, and a **random** knockout bracket; the dashboard's `P(advance)` is a fabricated placeholder. This work makes `wc2026_fixtures.csv` the single source of truth so the output is a genuine tournament prediction:

- Groups and group matches come from the CSV.
- The Round of 32 follows the CSV's official slot pairings; R16→Final follow a deterministic bracket.
- Group-advancement probabilities are tallied from the simulation (no placeholders).
- A locked **pre-tournament** snapshot (dated `2026-06-10`) is stored, and later snapshots condition on real results as each round completes.
- The dashboard shows a **probabilistic bracket**: the fixed knockout structure with each match annotated by the teams most likely to reach it.

---

## 2. Current state & key findings

- **Group rows (1–72): usable.** 48 teams across Groups A–L. Names need light normalization; all teams have Elo ratings except Curaçao (neutral fallback).
- **Round of 32 (rows 73–88): correct.** FIFA slot grammar: `Winner C`, `Runner-up F`, `3rd Place (A/B/C/D/F)`.
- **R16→Final (rows 89–104): malformed.** References (`M89 Winner`) point at the R16 matches themselves and are duplicated. The real bracket tree past R32 **cannot** be reconstructed from the CSV. Decision: ignore those references and build R16→Final by **standard adjacency** over the CSV's R32 order.

---

## 3. Team normalization & confederations

A `fixtures_bracket.py` module owns:

```python
TEAM_NAME_MAP = {
    "USA": "United States",
    "Turkiye": "Turkey",
    "Curaçao": "Curacao",
}
```

Normalization is applied to every team read from the fixtures file before lookups against Elo/features (which use the canonical names). Unmapped names pass through unchanged.

The 48-team confederation map (extends/overrides the existing `CONFEDERATION` in `feature_builder.py`):

| Confederation | Teams (canonical names) |
|---|---|
| UEFA | Czechia, Switzerland, Bosnia and Herzegovina, Scotland, Turkey, Germany, Netherlands, Sweden, Belgium, Spain, France, Norway, Austria, Portugal, Croatia, England |
| CONMEBOL | Brazil, Paraguay, Ecuador, Uruguay, Argentina, Colombia |
| CAF | South Africa, Morocco, Ivory Coast, Tunisia, Egypt, Cape Verde, Senegal, Algeria, DR Congo, Ghana |
| AFC | South Korea, Qatar, Australia, Japan, Iran, Saudi Arabia, Iraq, Jordan, Uzbekistan |
| CONCACAF | Mexico, Canada, Haiti, Curacao, United States, Panama |
| OFC | New Zealand |

(Single source of truth: derive group membership from the CSV at load time; the confederation map is a lookup keyed by canonical name.)

---

## 4. Module: `src/fixtures_bracket.py`

Parses the fixtures file into a structured, testable tournament definition. No model code.

```python
@dataclass
class Tournament:
    groups: dict[str, list[str]]              # "A" -> [4 canonical team names]
    group_matches: list[tuple[str, str, str]] # (team_a, team_b, group)
    r32_slots: list[tuple[str, str]]          # 16 pairs of slot strings (rows 73-88)

def load_tournament(path="data/raw/wc2026_fixtures.csv") -> Tournament: ...
def normalize_team(name: str) -> str: ...
def parse_slot(slot: str) -> Slot: ...   # "Winner C" / "Runner-up F" / "3rd Place (A/B/C/D/F)"
```

**Slot grammar** parsed into a typed form:
- `Winner <G>` → group winner of G
- `Runner-up <G>` → group runner-up of G
- `3rd Place (<G1>/<G2>/.../<Gn>)` → one of the 8 best third-place teams, eligible groups = the listed set

Only the 16 R32 rows are parsed into slots; R16→Final rows are ignored (see §2).

---

## 5. Third-place assignment

12 groups produce 12 third-place teams; the **8 best** advance (by pts, then GD, then GF — existing `get_8_best_third`). Each R32 third-place slot lists its eligible groups. We assign the 8 qualifying third-place teams to the 8 slots using **FIFA's official combination table**: the set of which-groups-qualified maps deterministically to a slot assignment. Implementation:

- Hardcode the official lookup `dict[frozenset(group_letters) -> dict[slot_index -> group_letter]]` for the standard combinations.
- If the realized combination is missing from the table (should not happen for valid data), fall back to a constrained greedy/bipartite match that respects each slot's eligible-group set.

This is internal to the simulator's per-tournament resolution.

---

## 6. Matchup features: `build_matchup_features(team_a, team_b, as_of_date)`

Knockout pairings are dynamic and absent from the fixtures file, so the simulator currently scores them with stale/neutral features. New helper in `feature_builder.py`:

```python
def build_matchup_features(
    team_a, team_b, as_of_date, elo_history, results, shootouts
) -> pd.DataFrame:  # single-row, FEATURE_COLS
```

Computes the 18 features from each team's **latest state as of `as_of_date`** (Elo, derived rank, rolling form/goals, head-to-head, rest days, confederation one-hots, penalty rate; `squad_value_ratio=0.0`). Poisson features (`lambda_a`, `p_win_poisson`) are appended via the existing Poisson model.

The simulator precomputes probabilities for **all 48×47 ordered pairs** using this helper (one batched `predict_proba`), so every group and knockout matchup uses real current-state features. `as_of_date` defaults to the snapshot's `as_of_date`.

---

## 7. Simulator changes (`src/simulator.py`)

`TournamentSimulator` is constructed from a `Tournament` plus a model, penalty rates, and a matchup-feature provider:

```python
TournamentSimulator(model, penalty_rates, tournament, matchup_features, known_results=None)
```

- **Groups:** play `tournament.group_matches` (the real fixtures), accumulate standings, sort by pts/GD/GF.
- **R32:** resolve each `r32_slots` pair → Winner/Runner-up via standings, third-place via §5 assignment.
- **R16→Final:** standard adjacency over R32 order — `winner(73)·winner(74) → R16`, etc., through QF, SF, Final.
- **Per-simulation tallies:** champion, finalist, semifinalist, **group_winner, runner_up, advanced** (reached R32).
- **`run()` output columns:** `team, p_champion, p_finalist, p_semifinalist, p_group_winner, p_runner_up, p_advance, confederation`.
- Keeps the matchup-probability cache (precompute once, then zero model calls in the loop) and the `progress_callback`.

### Known-results override

`known_results` (default `None`) is a mapping of already-decided matches to outcomes. For any match whose participants/slot are present, the recorded result is used verbatim instead of being simulated. Built from the actuals store (§8). This is the mechanism for conditioning later snapshots on reality; with `None`, everything is simulated (the pre-tournament baseline).

### Bracket-position occupancy

The knockout tree is a fixed set of positions: 16 R32 matches → 8 R16 → 4 QF → 2 SF → 1 Final (positions identified by `(round, match_index)` via the deterministic adjacency, so position is stable across simulations). Each simulation, the simulator records which two teams occupy each match position. Across all runs this yields, per position, the probability each team appears there — i.e. its probability of *reaching* that match.

`run()` returns this as a second result alongside the per-team table: a long-form bracket frame with columns `round, match_index, team, p_reach` (probability that team is one of the two participants in that match). Only this aggregate is kept — individual simulated brackets are not stored.

---

## 8. Prediction snapshots & actual results

**Snapshot store:** `data/processed/predictions/`
- `save_prediction_snapshot(team_df, bracket_df, as_of_date, label)` → writes the per-team table `{as_of_date}__{label}.csv` **and** the bracket frame `{as_of_date}__{label}__bracket.csv`, and appends to `predictions/index.csv` (`as_of_date, label, n_simulations, generated_at, path, bracket_path`).
- The newest snapshot's two files are also copied to `data/processed/simulation_results.csv` and `data/processed/bracket.csv` (the dashboard's defaults).
- **Immutability:** an existing snapshot file is never overwritten; re-running the same `as_of_date/label` requires `--force`.

**Locked baseline:** first run produces `2026-06-10__pre_tournament.csv` with `known_results=None`.

**Actuals store:** `data/raw/wc2026_actual_results.csv`, columns `stage, team_a, team_b, score_a, score_b, winner`. Starts empty; the user fills in each round's outcomes as they complete. A loader (`load_actual_results`) converts it into the `known_results` override (canonical names via `normalize_team`).

---

## 9. Pipeline & CLI

The `simulate` stage gains parameters and CLI flags:

```
python -m src.pipeline --start-from simulate --as-of 2026-06-10 --label pre_tournament
python -m src.pipeline --start-from simulate --as-of 2026-06-28 --label after_groups
```

- `--as-of DATE` (default `2026-06-10`) — snapshot date and `as_of_date` for matchup features.
- `--label TEXT` (default `pre_tournament`) — snapshot label.
- The stage loads the actuals file (if present and non-empty) into `known_results`, simulates the rest, saves the snapshot, and updates `simulation_results.csv`.
- Earlier snapshots remain untouched.

---

## 10. Dashboard

- **Group Stage Standings:** show real `p_advance` (and `p_group_winner`) per team from the snapshot — placeholder formula removed.
- **Tournament Bracket (new tab):** renders the fixed knockout structure (R32 → R16 → QF → SF → Final) as a left-to-right bracket of match boxes. Each box lists the **top-N** teams (default 3) most likely to occupy that match, with their `p_reach` percentages, read from the snapshot's bracket frame. Built with Plotly shapes + annotations (or an HTML/SVG layout); no model code at runtime. A `top_n` control adjusts how many teams show per box.
- **Snapshot selector:** a control to pick which snapshot to view (from `predictions/index.csv`), defaulting to the latest; drives both the per-team views and the bracket. Option to compare a snapshot's `p_champion` against the `pre_tournament` baseline.
- Champion heatmap and other tabs read CSV-derived groups, staying consistent.

---

## 11. Testing

- **Parser:** group derivation (12 groups × 4), name normalization (`USA`→`United States`, etc.), R32 slot parsing (winner/runner-up/third-place with eligible sets).
- **Third-place assignment:** every assigned team's group is within its slot's eligible set; 8 slots filled.
- **Bracket:** adjacency reduces 32→1; one champion; finalists/semifinalists populated.
- **Bracket occupancy:** frame has the right positions per round (16/8/4/2/1); for each `(round, match_index)` the team `p_reach` values sum to ~2 (two participants per match); Final `p_reach` per team equals its `p_finalist`.
- **Advancement tallies:** `p_advance` ≈ `p_group_winner + p_runner_up + best-third share`; probabilities in [0,1]; champion probs sum to ~1.
- **Known-results override:** a pinned match always yields the recorded winner; pinned group results fix that group's standings.
- **Snapshots:** `save_prediction_snapshot` writes the file + index row; existing snapshot not overwritten without force.
- **Matchup features:** returns one row with all `FEATURE_COLS`; uses `as_of_date` state.

---

## 12. Out of scope

- Automated fetching of real results (manual CSV entry only).
- Correcting the malformed R16→Final references in the source CSV (we ignore them by design).
- Re-training models per snapshot (snapshots re-simulate with fixed models; retraining stays a separate `--force` pipeline run).
- Squad value integration (`squad_value_ratio` stays `0.0`).
