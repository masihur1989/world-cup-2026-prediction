# World Cup 2026 Prediction System — Complete Guide

This guide explains everything that's built, how to run each feature, and how to keep the
predictions up to date as the tournament plays out.

- [1. What's built](#1-whats-built)
- [2. Architecture & pipeline stages](#2-architecture--pipeline-stages)
- [3. Setup](#3-setup)
- [4. Data files](#4-data-files)
- [5. Running the pipeline](#5-running-the-pipeline)
- [6. The dashboard](#6-the-dashboard)
- [7. Updating the data](#7-updating-the-data)
- [8. Verifying model quality](#8-verifying-model-quality)
- [9. Testing](#9-testing)
- [10. Known limitations](#10-known-limitations)

---

## 1. What's built

A hybrid model that predicts any World Cup 2026 match and simulates the whole tournament. It
runs **entirely on local CSV data** — no external APIs, no scraping.

| Layer | What it does |
|-------|--------------|
| **Stage 1 — Poisson GLM** | Expected goals (λ) per team, via `statsmodels` |
| **Stage 2 — XGBoost** | Calibrated Win/Draw/Loss probabilities (3-class for groups, binary for knockouts); Optuna-tuned, `TimeSeriesSplit` CV, SHAP, MLflow logging |
| **Stage 3 — Monte Carlo simulator** | 100k tournament runs driven by the real fixtures → champion / advancement / bracket probabilities |
| **Snapshots** | Each run is saved as a dated, immutable snapshot; results can be updated round-by-round with real outcomes |
| **Dashboard** | A 4-tab Streamlit UI that reads pre-computed snapshots (no model code at runtime) |

The simulator is driven by `data/raw/wc2026_fixtures.csv`: real groups, the official Round-of-32
slot pairings, and standard-adjacency R16 → Final.

---

## 2. Architecture & pipeline stages

Strictly sequential. Each stage writes a checkpoint artifact to `data/processed/`, so a failed
run resumes instead of restarting.

```
data/raw/*.csv
  → src/data_loader.py        load + filter to 1990+
  → src/elo_calculator.py     Elo from scratch          → elo_history.csv
  → src/feature_builder.py    18 leak-free features      → features.csv (+ poisson_model.pkl)
  → src/xgboost_model.py       3-class + binary models    → xgb_3class.pkl, xgb_binary.pkl
  → (SHAP)                                                 → shap_values.csv
  → src/simulator.py           fixture-driven Monte Carlo → snapshot + bracket + matchups
  → app/dashboard.py           reads data/processed/ only
```

| # | Stage | Checkpoint artifact |
|---|-------|---------------------|
| 1 | `elo` | `elo_history.csv` |
| 2 | `features` | `features.csv` (+ `poisson_model.pkl`) |
| 3 | `xgb_3class` | `xgb_3class.pkl` |
| 4 | `xgb_binary` | `xgb_binary.pkl` |
| 5 | `shap` | `shap_values.csv` |
| 6 | `simulate` | `simulation_results.csv` (+ snapshot, bracket, matchups) |

Supporting modules: `src/fixtures_bracket.py` (parses the fixtures CSV into groups / matches /
R32 slots, normalizes team names, assigns third-place teams), `src/predictions.py` (snapshot
store + actual-results loader), `src/pipeline.py` (orchestration + CLI).

---

## 3. Setup

Requires Python 3.11+ (developed on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt   # full stack (pipeline + training + tests)
```

`requirements.txt` is the **slim dashboard runtime** (pandas, numpy, streamlit, plotly) used by
the deployed app; `requirements-train.txt` adds the training stack (xgboost, statsmodels, optuna,
shap, scikit-learn, mlflow, pytest). To run *only* the dashboard, `requirements.txt` suffices.

> `requirements-train.txt` pins `setuptools<81` — `shap`/`numba` still import `pkg_resources`,
> removed in setuptools 81+. **Always use the venv** (`.venv/bin/...`), not a base Anaconda install.

---

## 4. Data files

### Inputs — `data/raw/`

| File | Committed? | Purpose |
|------|-----------|---------|
| `wc2026_fixtures.csv` | ✅ | Groups, 72 group matches, R32 slot pairings (source of truth for the tournament structure) |
| `wc2026_actual_results.csv` | ✅ (header only) | Real match outcomes you fill in as the tournament progresses |
| `results.csv` | ❌ download | Historical international results 1872–2024 (filtered to 1990+) |
| `shootouts.csv` | ❌ download | Penalty shootout winners |
| `goalscorers.csv` | ❌ download | Scorer data (reserved) |

The three downloaded files come from Kaggle:
**[martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)**.
Place them in `data/raw/`.

### Outputs — `data/processed/` (gitignored, except committed snapshots)

- `elo_history.csv`, `features.csv`, `*.pkl` models, `shap_values.csv`
- `simulation_results.csv`, `bracket.csv`, `matchups.csv` — "latest" files the dashboard falls back to
- `predictions/` — dated snapshots:
  - `{date}__{label}.csv` — per-team probabilities
  - `{date}__{label}__bracket.csv` — per-bracket-position reach probabilities
  - `{date}__{label}__matchups.csv` — per-pair W/D/L probabilities
  - `index.csv` — registry the dashboard's snapshot selector reads

The locked baseline `2026-06-10__pre_tournament` is committed.

---

## 5. Running the pipeline

All commands assume the venv is active (or prefix with `.venv/bin/`).

**Full run** (needs the Kaggle CSVs in `data/raw/`):

```bash
python -m src.pipeline --trials 100 --simulations 100000
```

Logs timestamped per-stage progress and `~5%` updates inside long loops.

**Check status** — which stages are done, what runs next:

```bash
python -m src.pipeline --status
```

**Resume after a failure** — automatically skips completed stages:

```bash
python -m src.pipeline
```

**Re-run from a specific stage** (keeps earlier checkpoints):

```bash
python -m src.pipeline --start-from xgb_3class
```

**Force a clean run from scratch** (ignores all checkpoints — retrains everything):

```bash
python -m src.pipeline --force
```

> ⚠️ `--force` overrides `--start-from` and reruns **all** stages (the ~20-minute full run,
> including Optuna). To regenerate only the simulation, use `--start-from simulate` **without**
> `--force` (delete the target snapshot first if it already exists, or pick a new `--label`).

**Snapshot options** (apply to the `simulate` stage):

```bash
python -m src.pipeline --start-from simulate --as-of 2026-06-10 --label pre_tournament
```

- `--as-of DATE` — snapshot date + the as-of date used for team form/Elo features
- `--label TEXT` — snapshot name
- Re-running an existing `date/label` requires `--force`.

**Performance:** the simulator scores every distinct matchup once and caches it, so a 100k-run
simulation finishes in ~30 seconds. Use `--simulations 10000` for faster, still-stable estimates.

**Smoke test** (no Kaggle data — runs every stage on synthetic data):

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

---

## 6. The dashboard

```bash
streamlit run app/dashboard.py        # http://localhost:8501
```

Reads only `data/processed/` — no model code runs. A **sidebar snapshot selector** chooses which
saved snapshot all tabs display. Team names show **emoji flags** throughout.

| Tab | What it shows |
|-----|---------------|
| **Match Predictor** | Real model W/D/L probabilities. Two modes: **Scheduled fixture** (pick from the 72 group matches) or **Any matchup** (any two teams, e.g. a hypothetical final). Pure lookup from the snapshot's matchup table. |
| **Champion Probabilities** | 12-group × 4-team heatmap shaded by P(champion). |
| **Group Stage Standings** | Real P(advance) and P(win group) per team, per group. |
| **Tournament Bracket** | Probabilistic bracket: R32 → Final, each match box listing the top-N teams most likely to reach it (top-N slider). |

(Model interpretability via SHAP is still computed during training and saved to
`data/processed/shap_values.csv`; it's no longer surfaced as a user-facing dashboard tab.)

---

## 7. Updating the data

### A. Update match results as the tournament plays out

This is the main ongoing workflow.

1. **Enter completed matches** in `data/raw/wc2026_actual_results.csv`:

   ```csv
   stage,team_a,team_b,score_a,score_b,winner
   Group A,Mexico,South Africa,2,1,Mexico
   Group A,South Korea,Czechia,0,0,
   Round of 32,Brazil,Norway,,,Brazil
   ```

   - `team_a`/`team_b` — fixtures-file spellings are fine (`USA`, `Turkiye`, `Curaçao` are auto-normalized).
   - **Group matches:** fill `score_a`/`score_b` (standings are computed from scores; leave `winner` blank for a draw).
   - **Knockout matches:** `winner` is what matters; scores can be blank.
   - Listed matches are **pinned** (used as-is); everything else is still simulated.

2. **Generate a new snapshot** for the remaining matches:

   ```bash
   python -m src.pipeline --start-from simulate --as-of 2026-06-28 --label after_groups
   ```

3. **View it** — refresh the dashboard and pick the new snapshot from the sidebar. Earlier
   snapshots (including the locked baseline) stay immutable, so you can compare how the forecast
   shifts round by round.

### B. Update the fixtures (e.g. corrected draw or schedule)

Edit `data/raw/wc2026_fixtures.csv` (columns: `team_a,team_b,stage,date,venue`). Keep the
`Group X` rows and the `Round of 32` slot grammar (`Winner C`, `Runner-up F`,
`3rd Place (A/B/C/D/F)`). If you add teams not already known, add them to the `CONFEDERATION`
map in `src/feature_builder.py` and (for flags) `TEAM_ISO2` in `app/dashboard.py`. Then rebuild:

```bash
python -m src.pipeline --start-from features --as-of 2026-06-10 --label pre_tournament --force
```

### C. Refresh the historical training data

Download an updated `results.csv` / `shootouts.csv` from Kaggle into `data/raw/`, then retrain:

```bash
python -m src.pipeline --force        # full rebuild incl. Elo, features, models, simulation
```

---

## 8. Verifying model quality

Training logs metrics to MLflow on the WC 2022 holdout. Inspect them:

```bash
python -c "
import mlflow
c = mlflow.tracking.MlflowClient()
exp = c.get_experiment_by_name('wc2026')
for r in c.search_runs(exp.experiment_id, order_by=['start_time DESC'])[:3]:
    print(r.data.tags.get('mlflow.runName'), r.data.metrics)
"
```

Or browse: `mlflow ui` → http://localhost:5000.

Targets (3-class model): accuracy > 0.55, log-loss < 0.95, Brier < 0.22, RPS < 0.20. See
[§10](#10-known-limitations) for how the current model compares.

---

## 9. Testing

```bash
python -m pytest tests/ -q          # full suite (~78 tests)
```

Some tests need the trained model artifacts in `data/processed/` and skip without them.

---

## 10. Known limitations

- **Holdout metrics fall short of targets.** The 3-class model scores ~0.52 accuracy / ~1.0
  log-loss on the small (64-match) WC 2022 holdout — below the spec targets. Treat the 2026
  numbers as **directionally reliable (rankings, advancement odds)** rather than precise to the
  decimal. Levers: more Optuna trials, a wider training window, real squad-value features.
- **`squad_value_ratio` is hardcoded `0.0`** (placeholder for a future Transfermarkt feed).
- **Knockout bracket past R32** uses standard adjacency over the CSV's R32 order (the fixtures
  file's R16+ references are malformed and ignored by design).
- **Third-place assignment** uses constrained matching against each slot's eligible groups
  (equivalent to FIFA's official combination table for valid data, without hardcoding it).
- **Curaçao** has no Elo history → near-neutral predictions for its matches.
- **On some Windows setups** emoji flags render as letter pairs (OS font limitation).

---

For the original design rationale see `docs/superpowers/specs/`; for the build plans see
`docs/superpowers/plans/`.
