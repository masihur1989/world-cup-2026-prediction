# ML Model Build-Out

A technical account of the modeling work behind the World Cup 2026 prediction system: the data,
the features, each of the three model stages, how they're trained and evaluated, and the design
decisions and trade-offs made along the way.

- [1. Problem framing](#1-problem-framing)
- [2. Data](#2-data)
- [3. Elo ratings (from scratch)](#3-elo-ratings-from-scratch)
- [4. Feature engineering](#4-feature-engineering)
- [5. Stage 1 — Poisson GLM](#5-stage-1--poisson-glm)
- [6. Stage 2 — XGBoost classifier](#6-stage-2--xgboost-classifier)
- [7. Calibration & cross-validation](#7-calibration--cross-validation)
- [8. Evaluation](#8-evaluation)
- [9. Stage 3 — Monte Carlo simulator](#9-stage-3--monte-carlo-simulator)
- [10. Leakage prevention](#10-leakage-prevention)
- [11. Design decisions & trade-offs](#11-design-decisions--trade-offs)
- [12. Limitations & future work](#12-limitations--future-work)

---

## 1. Problem framing

The core prediction is a **single-match outcome** — Win / Draw / Loss from Team A's
perspective. That base predictor is then composed into a **full-tournament forecast** via Monte
Carlo simulation. Three stages, each feeding the next:

1. **Poisson GLM** → expected goals (λ_A, λ_B) and a Poisson-derived win probability.
2. **XGBoost** → calibrated W/D/L probabilities (using the Poisson outputs as features).
3. **Monte Carlo** → champion / advancement / bracket probabilities over 100k simulated tournaments.

Hybrid by design: the Poisson stage injects a goal-scoring prior, XGBoost captures non-linear
interactions and calibrates probabilities, and the simulator turns per-match probabilities into
tournament-level outcomes.

---

## 2. Data

Source: the public Kaggle dataset
**martj42/international-football-results-from-1872-to-2017** (`results.csv`, `shootouts.csv`,
`goalscorers.csv`), plus a hand-built `wc2026_fixtures.csv`.

- **Filtering:** `results.csv` is filtered to `date >= 1990-01-01`.
- **Train split:** matches with `date <= 2018-12-31`.
- **Validation (holdout):** `tournament == 'FIFA World Cup'` and `2019-01-01 ≤ date ≤ 2022-12-31`
  — i.e. **WC 2022** (64 group-stage matches in the 3-class holdout).

No external APIs or scraping — everything is computed from these local CSVs.

---

## 3. Elo ratings (from scratch)

Rather than importing published Elo numbers, ratings are computed directly from match history
(`src/elo_calculator.py`), giving full control and reproducibility.

- **Initialization:** every team starts at **1500** on 1990-01-01.
- **Chronological pass** through all matches, updating after each.
- **K-factor by tournament importance:**
  - `40` — FIFA World Cup
  - `30` — continental tournaments + their qualifiers (Euro, Copa América, AFCON, Asian Cup, Gold Cup, OFC Nations Cup)
  - `20` — everything else (friendlies, etc.)
- **Update rule:**
  ```
  expected_A = 1 / (1 + 10^((elo_B − elo_A) / 400))
  elo_A     += K · (actual_A − expected_A)        # actual ∈ {1 win, 0.5 draw, 0 loss}
  ```
- **Derived FIFA-style rank:** on any date, "active" teams (≥1 match in the prior 12 months) are
  ordered by current Elo; rank 1 = highest. This feeds `fifa_rank_diff` without any official
  ranking file.

`elo_history.csv` stores `(date, team, elo_rating)` after every match.

---

## 4. Feature engineering

18 features per match (`src/feature_builder.py`), all **pre-match** (computed only from
information available before kickoff).

| # | Feature | Definition |
|---|---------|-----------|
| 1 | `elo_diff` | Elo_A − Elo_B (as of match date) |
| 2 | `fifa_rank_diff` | rank_B − rank_A (derived Elo rank) |
| 3–4 | `form_A`, `form_B` | rolling sum of points (W=3, D=1, L=0) over last 10 matches |
| 5–6 | `goals_scored_avg_A/B` | decay-weighted mean goals scored over last 6 (weights 0.5→1.0) |
| 7–8 | `goals_conceded_avg_A/B` | decay-weighted mean goals conceded over last 6 |
| 9 | `h2h_win_rate_A` | A's historical win rate vs B (draws = 0.5) |
| 10 | `h2h_goal_diff` | A's mean goal margin in prior A-vs-B meetings |
| 11 | `squad_value_ratio` | **hardcoded 0.0** (placeholder for Transfermarkt data) |
| 12 | `stage_weight` | 1 Group … 5 Final (knockout importance) |
| 13 | `rest_days_diff` | days since last match, A − B |
| 14–25 | `conf_{A,B}_{UEFA,CONMEBOL,CAF,AFC,CONCACAF,OFC}` | confederation one-hots (12 cols) |
| 26 | `penalty_win_rate_A` | A's historical shootout win rate |
| 27 | `lambda_a` | Stage-1 Poisson expected goals for A |
| 28 | `p_win_poisson` | Stage-1 Poisson win probability for A |

**Team-perspective construction:** each match is expanded into two rows (one per team) via
`build_team_perspective`, then rolling stats are computed per team with `shift(1)` so the
current match never contributes to its own features.

`FEATURE_COLS` (28 columns) = 13 scalar + 12 confederation one-hots + `penalty_win_rate_A` +
`lambda_a` + `p_win_poisson`.

---

## 5. Stage 1 — Poisson GLM

Two `statsmodels` GLMs with a Poisson family and log link (`src/poisson_model.py`), one per team
perspective, predicting expected goals:

- **Model A features:** `elo_diff, form_A, goals_scored_avg_A, fifa_rank_diff, h2h_goal_diff, stage_weight`
- **Model B features:** the same with **negated differentials** (`elo_diff_neg`, `fifa_rank_diff_neg`, `h2h_goal_diff_neg`) and Team-B stats.

Targets are actual goals scored (`home_score`/`away_score`), falling back to rolling goal
averages where scores are absent. Predicted λ is floored at 0.1 to stay positive.

**Scoreline simulation:** given (λ_A, λ_B), 50,000 independent Poisson draws per side yield
`p_win_A`, `p_draw`, `p_win_B`. `lambda_a` and `p_win_poisson` become features 27–28, passing a
goal-scoring prior into Stage 2.

---

## 6. Stage 2 — XGBoost classifier

Two `XGBClassifier` variants (`src/xgboost_model.py`):

- **3-class** (`mode="3class"`) — Win / Draw / Loss, used for group matches.
- **Binary** (`mode="binary"`) — Win / Loss only (draw suppressed), used for knockouts.

Labels are encoded with `LabelEncoder` ({−1, 0, 1}). Modern XGBoost infers `objective`/
`num_class`, so those aren't passed explicitly (avoids version pitfalls); `tree_method="hist"`.

**Hyperparameter tuning — Optuna, 100 trials**, minimizing mean cross-validated log-loss over a
search space of:

| Param | Range |
|-------|-------|
| `max_depth` | 3–8 |
| `learning_rate` | 0.01–0.3 (log scale) |
| `n_estimators` | 100–600 |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `scale_pos_weight` | 0.5–3.0 (draw-class balancing) |

Every trial's params and the final metrics are logged to **MLflow** (experiment `wc2026`).

**Interpretability:** SHAP `TreeExplainer` on the underlying booster produces per-feature
attributions; the pipeline saves mean |SHAP| per feature to `shap_values.csv` for the dashboard.

---

## 7. Calibration & cross-validation

- **Temporal CV only:** `TimeSeriesSplit(n_splits=5)` after sorting by date — never a random
  shuffle, since match data is time-ordered and a shuffle would leak the future.
- **Probability calibration:** the tuned booster is wrapped in
  `CalibratedClassifierCV(method="sigmoid", cv=TimeSeriesSplit(5))`. Platt scaling corrects the
  raw scores and guarantees the output probabilities **sum to 1.0** — essential because the
  simulator samples from them.

---

## 8. Evaluation

Four metrics on the WC 2022 holdout:

- **Accuracy** — argmax correct.
- **Log-loss** — penalizes confident wrong calls.
- **Brier** — mean squared error across the one-hot class vector.
- **RPS (Ranked Probability Score)** — ordinal-aware (W > D > L), the fairest single number for a 3-class football outcome.

| Metric | Target (3-class) | 3-class actual | Binary actual |
|--------|------------------|----------------|---------------|
| Accuracy | > 0.55 | 0.516 | 0.673 |
| Log-loss | < 0.95 | 1.017 | 0.640 |
| Brier | < 0.22 | 0.603 | 0.441 |
| RPS | < 0.20 | 0.214 | 0.220 |

**Reading the numbers honestly:** the 3-class model lands just under the accuracy target and
modestly over on log-loss, with RPS near the line. The Brier target (<0.22) was effectively
mis-specified for a 3-class one-hot Brier (range 0–2), so it isn't a fair gate as written. The
model has **real discrimination** (51.6% on a 3-way task vs ~33% random) but **imperfect
calibration**; the holdout is also small (64 matches), so the metrics carry a few points of
noise. Practical implication: trust the *ranking* of teams more than the exact percentages.

---

## 9. Stage 3 — Monte Carlo simulator

`src/simulator.py`, driven entirely by the parsed fixtures (`src/fixtures_bracket.py`).

**Current-state matchup features.** Knockout pairings are dynamic (any team can meet any team),
so a `MatchupFeatureProvider` builds the 18 features for any pair using each team's **latest
state as of the snapshot date** — Elo, form, goals, head-to-head, rest, confederation, penalty
rate — then appends the Poisson features. This makes knockout odds use real features, not stale
historical rows.

**Probability cache (key optimization).** There are only ~2,256 distinct ordered pairs among 48
teams, versus ~10M `predict_proba` calls if scored per simulated match. The simulator scores all
pairs **once** (`precompute_probabilities`) and the simulation loop is then pure sampling — a
100k-tournament run completes in ~30 seconds. The same cache is exported as `matchup_table()` for
the dashboard's Match Predictor.

**Tournament logic per simulation:**
- **Groups:** play the real fixtures, accumulate points/GD/GF, rank each group.
- **Advancement:** top 2 per group + the **8 best third-place** teams, assigned to the R32
  third-place slots by constrained matching against each slot's eligible-group set.
- **Round of 32:** resolved from the fixtures' official slot pairings.
- **R16 → Final:** standard single-elimination adjacency over the R32 order.
- **Knockout ties:** when the model is ~50/50, broken by historical penalty shootout win rate
  (fallback 50/50).

**Aggregated over 100k runs:** per-team `p_champion`, `p_finalist`, `p_semifinalist`,
`p_group_winner`, `p_runner_up`, `p_advance`; per-bracket-position reach probabilities; and the
per-pair matchup table.

**Conditioning on reality.** A `known_results` override (loaded from
`wc2026_actual_results.csv`) pins already-played matches to their real outcomes; everything else
is simulated. This drives the round-by-round snapshot updates.

---

## 10. Leakage prevention

Temporal leakage is the main risk in sports modeling. Guards applied throughout:

- **`shift(1)` / expanding windows** for all rolling features — a match never sees its own result.
- **Elo** is computed in a strict chronological pass; a match's features use the rating *before* it.
- **Head-to-head** uses only meetings with `date < match_date`.
- **`TimeSeriesSplit`** for both tuning and calibration — no random shuffles.
- **Train/validation split is purely temporal** (≤2018 train, WC 2022 holdout).

---

## 11. Design decisions & trade-offs

- **Elo from scratch** instead of published ratings → reproducible, no external dependency; cost
  is that it's a simpler model than the official multi-factor systems.
- **Poisson features feeding XGBoost** → combines a goal-scoring generative prior with a
  discriminative calibrated classifier, rather than choosing one.
- **Sigmoid calibration** → probabilities are sound to sample from, at a small cost to raw
  sharpness.
- **Matchup-probability cache** → ~100× simulation speedup with identical statistics.
- **Constrained third-place matching** instead of hardcoding FIFA's 495-row combination table →
  equivalent valid output for real data, far less transcription risk.
- **Standard bracket adjacency past R32** → the fixtures file's R16+ references are malformed, so
  adjacency over the (correct) R32 order is the deterministic, defensible choice.

---

## 12. Limitations & future work

- **Calibration is the weakest link.** Log-loss above target and a noisy 64-match holdout mean
  exact percentages should be treated as indicative. Levers: more Optuna trials, a wider training
  window (include 2019–2021 qualifiers), and probability recalibration.
- **`squad_value_ratio = 0.0`** — integrating real Transfermarkt squad values is the most
  promising missing feature.
- **Sparse-history teams** (e.g. Curaçao) get near-neutral predictions.
- **Knockout perspective** — the Match Predictor shows the group-stage W/D/L view; knockout
  tie-breaking lives in the simulator, not in that displayed probability.
- **Goal-scorer data** (`goalscorers.csv`) is loaded but not yet used — a route to player-level
  enrichment.

---

For how to run training and inspect metrics, see [USAGE.md](USAGE.md). For the original design
rationale and the task-by-task build, see `docs/superpowers/specs/` and `docs/superpowers/plans/`.
