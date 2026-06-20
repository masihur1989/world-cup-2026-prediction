# World Cup 2026 Match Outcome Prediction System — Design Spec

**Date:** 2026-06-20  
**Status:** Approved  
**Scope:** Full hybrid prediction system — Poisson GLM + XGBoost + Monte Carlo simulator + Streamlit dashboard

---

## 1. Objective

Predict the winner of any FIFA World Cup 2026 match using a three-stage hybrid model:

- **Stage 1:** Poisson Regression → expected goals (λ_A, λ_B) per team
- **Stage 2:** XGBoost Classifier → W/D/L probabilities
- **Stage 3:** Monte Carlo Tournament Simulator → champion probabilities

**Test strategy:** WC 2022 (2019-01-01 – 2022-12-31) is the holdout. Completed WC 2026 matches are ignored. `wc2026_fixtures.csv` contains only remaining/future fixtures.

---

## 2. Data Sources (local CSV only — zero external API calls)

| File | Source | Description |
|------|--------|-------------|
| `data/raw/results.csv` | Kaggle: martj42/international-football-results-from-1872-to-2017 | Match results 1872–2024; filtered to 1990+ for training |
| `data/raw/goalscorers.csv` | Same dataset | Scorer-level data; used for future enrichment |
| `data/raw/shootouts.csv` | Same dataset | Penalty shootout winners |
| `data/raw/wc2026_fixtures.csv` | Manually created | Remaining WC 2026 fixtures only |

**Key filters:**
- Training: `results.csv` where `date >= 1990-01-01`
- WC-only subset: `tournament == 'FIFA World Cup'`
- Train split: 1990–2018 (WC + qualifiers)
- Validation split: `tournament == 'FIFA World Cup'` AND `2019-01-01 <= date <= 2022-12-31`

**wc2026_fixtures.csv columns:** `team_a, team_b, stage, date, venue`

---

## 3. Architecture — Sequential Pipeline (Option A)

Each stage runs in strict order and serializes outputs to `data/processed/` as CSVs. The Streamlit app reads only pre-computed files — no model code runs at dashboard runtime.

```
data/raw/ CSVs
    │
    ▼
src/data_loader.py      → filtered DataFrames in memory
    │
    ▼
src/elo_calculator.py   → data/processed/elo_history.csv
    │
    ▼
src/feature_builder.py  → data/processed/features.csv
    │
    ▼
src/poisson_model.py    → λ_A, λ_B, p_win_poisson (appended to features)
    │
    ▼
src/xgboost_model.py    → trained model artifacts + SHAP values
    │
    ▼
src/simulator.py        → data/processed/simulation_results.csv
    │
    ▼
app/dashboard.py        → Streamlit UI (reads processed/ only)
```

**Orchestration:** `src/pipeline.py` calls each stage in sequence. Each stage can also be run independently for debugging.

---

## 4. Project Structure

```
soccer_predictor/
├── data/
│   ├── raw/
│   │   ├── results.csv
│   │   ├── goalscorers.csv
│   │   ├── shootouts.csv
│   │   └── wc2026_fixtures.csv
│   └── processed/
│       ├── elo_history.csv
│       ├── features.csv
│       └── simulation_results.csv
├── src/
│   ├── data_loader.py
│   ├── elo_calculator.py
│   ├── feature_builder.py
│   ├── poisson_model.py
│   ├── xgboost_model.py
│   ├── simulator.py
│   └── pipeline.py
├── notebooks/
│   └── exploration.ipynb
├── app/
│   └── dashboard.py
├── mlruns/
├── docs/
│   └── superpowers/specs/
└── requirements.txt
```

---

## 5. Elo Rating System

- **Initialization:** all teams = 1500 on 1990-01-01
- **Chronological iteration** through `results.csv`
- **K-factors:** 40 (FIFA World Cup), 30 (continental tournaments), 20 (all others)
- **Update formula:**
  ```
  Expected = 1 / (1 + 10^((opponent_elo - team_elo) / 400))
  New Elo  = Old Elo + K × (Actual − Expected)
  Actual   = 1 (win), 0.5 (draw), 0 (loss)
  ```
- **Output:** `elo_history.csv` — columns: `date, team, elo_rating`
- **FIFA rank derivation:** on each match date, rank all active teams (≥1 match in prior 12 months) by Elo descending; rank 1 = best

---

## 6. Feature Engineering (18 features, all pre-match only)

All features computed using `groupby + shift` or expanding windows — never using future rows.

| # | Feature | Source |
|---|---------|--------|
| 1 | `elo_diff` | Team A Elo − Team B Elo (from elo_history) |
| 2 | `fifa_rank_diff` | derived rank B − derived rank A |
| 3 | `form_A` | weighted points, last 10 matches (W=3, D=1, L=0) |
| 4 | `form_B` | same for Team B |
| 5 | `goals_scored_avg_A` | rolling 6-match avg goals scored (decay-weighted) |
| 6 | `goals_scored_avg_B` | same for Team B |
| 7 | `goals_conceded_avg_A` | rolling 6-match avg goals conceded |
| 8 | `goals_conceded_avg_B` | same for Team B |
| 9 | `h2h_win_rate_A` | Team A historical win % vs Team B |
| 10 | `h2h_goal_diff` | avg goal margin across all A vs B meetings |
| 11 | `squad_value_ratio` | **hardcoded 0.0** — TODO: integrate Transfermarkt data |
| 12 | `stage_weight` | 1=Group, 2=R32, 3=QF, 4=SF, 5=Final |
| 13 | `rest_days_diff` | days since last match (A − B), from results.csv |
| 14–15 | `confederation_A/B` | one-hot: UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC |
| 16 | `penalty_win_rate_A` | Team A win rate in shootouts (from shootouts.csv) |
| 17 | `lambda_a` | output from Stage 1 Poisson model |
| 18 | `p_win_poisson` | P(Win A) from Monte Carlo on λ_A, λ_B |

**Confederation map:** hardcoded dict `{team: confederation}` covering all WC 2026 qualified nations.

---

## 7. Stage 1 — Poisson GLM

- **Library:** `statsmodels` GLM, Poisson family, log link
- **Two separate models** (one per team perspective)
- **Formula:**
  ```
  log(λ_A) = β0 + β1·elo_diff + β2·form_A + β3·goals_scored_avg_A
           + β4·rank_diff + β5·h2h_goal_diff + β6·stage_weight
  log(λ_B) = same with Team B features and negated differentials
  ```
- **Training data:** results.csv 1990–2018 (WC + qualifiers)
- **Simulation:** 50,000 scoreline draws via `scipy.stats.poisson.rvs` → derive `P(Win A)`, `P(Draw)`, `P(Win B)`
- **Outputs fed into Stage 2:** `lambda_a`, `lambda_b`, `p_win_poisson`

---

## 8. Stage 2 — XGBoost Classifier

**Target variable:** `outcome ∈ {1=Win A, 0=Draw, -1=Win B}`

**Two model variants:**
- **Group stage model:** 3-class (W/D/L)
- **Knockout model:** binary (W/L only, draw class suppressed)

**Hyperparameter tuning:**
- Optuna, 100 trials
- Search space: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `scale_pos_weight` (for draw class balancing)

**Calibration:** `CalibratedClassifierCV(method='sigmoid')` — ensures all probabilities sum to 1.0

**Cross-validation:** `TimeSeriesSplit(n_splits=5)` sorted by date — no random shuffle

**Interpretability:** SHAP values computed for every prediction

**Penalty shootout tiebreaker (knockout only):**
- Use `penalty_win_rate_A` from `shootouts.csv`
- Fallback to 50/50 if no prior shootout data for that team

**Validation split:** `tournament == 'FIFA World Cup'` AND `2019-01-01 ≤ date ≤ 2022-12-31`

**Target metrics on WC 2022 holdout:**
- 3-class accuracy > 55%
- Log-loss < 0.95
- Brier Score < 0.22
- RPS (Ranked Probability Score) < 0.20

**MLflow logging:** every model version, all Optuna trial params, and final metrics

---

## 9. Stage 3 — Monte Carlo Tournament Simulator

- **Simulations:** 100,000 full tournament runs
- **Input:** `wc2026_fixtures.csv` (remaining fixtures only)
- **Group stage logic:** round-robin → advance top 2 per group + 8 best 3rd-place teams (48 teams, 12 groups per WC 2026 format)
- **Knockout logic:** single-elimination, binary model, no draw class; penalty shootout via `penalty_win_rate_A`
- **Tracked per simulation:** champion, finalist, semi-finalists
- **Output:** `data/processed/simulation_results.csv`
  - Columns: `team, p_champion, p_finalist, p_semifinalist, confederation`

---

## 10. Streamlit Dashboard

Four views in `app/dashboard.py`. All data loaded from `data/processed/` at startup — no model code, no recomputation, no API calls.

| View | Content |
|------|---------|
| **Match Predictor** | Dropdown: any two WC 2026 teams → W/D/L probability bars (Plotly); Poisson vs XGBoost outputs side-by-side |
| **Champion Probabilities** | Horizontal bar chart sorted by `p_champion`; bars color-coded by confederation |
| **Group Stage Standings** | Table: expected points, goal diff, advancement probability per team per group (derived from simulation) |
| **SHAP Feature Importance** | `shap.plots.beeswarm` showing which of 18 features drive predictions |

**Confederation color map:**
- UEFA = blue, CONMEBOL = yellow, CAF = green, AFC = red, CONCACAF = orange, OFC = purple

---

## 11. Hard Constraints

- Zero external API calls — all data from `/data/raw/` CSV files only
- All features computed using only data available BEFORE each match date
- `TimeSeriesSplit` only — never random shuffle temporal data
- All probability outputs must sum to 1.0 (enforced by `CalibratedClassifierCV`)
- `squad_value_ratio = 0.0` with TODO comment for future Transfermarkt integration
- Knockout stage must suppress draw class and use binary classifier
- MLflow logs every model version, parameters, and metrics

---

## 12. Tech Stack

```
Python 3.11
pandas, numpy
statsmodels, scipy
xgboost, optuna, shap
scikit-learn (CalibratedClassifierCV, TimeSeriesSplit)
streamlit, plotly
mlflow
```

---

## 13. Out of Scope

- Live match data or real-time score updates
- Completed WC 2026 match tracking (future enhancement)
- Transfermarkt squad value integration (TODO placeholder only)
- External Elo or FIFA ranking APIs
- Any form of web scraping
