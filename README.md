# FIFA World Cup 2026 — Match Outcome Prediction System

A three-stage hybrid model that predicts the outcome of any World Cup 2026 match and
simulates the full tournament to estimate each team's championship probability. Everything
runs on **local CSV data only** — no external APIs, no web scraping.

| Stage | Component | Output |
|-------|-----------|--------|
| **1** | Poisson GLM (`statsmodels`) | Expected goals λ_A, λ_B per team |
| **2** | XGBoost classifier (3-class group / binary knockout) | Calibrated W/D/L probabilities |
| **3** | Monte Carlo tournament simulator | Champion / finalist / semifinalist probabilities |

A Streamlit dashboard reads the pre-computed results and presents four interactive views.

---

## Architecture

Strictly sequential pipeline. Each stage serializes its output so the dashboard never runs
model code at runtime.

```
data/raw/*.csv
   └─ src/data_loader.py      load + filter to 1990+
        └─ src/elo_calculator.py    Elo from scratch → elo_history.csv
             └─ src/feature_builder.py   18 pre-match features → features.csv
                  └─ src/poisson_model.py     λ_A, λ_B, p_win_poisson (features 17–18)
                       └─ src/xgboost_model.py     calibrated classifiers + SHAP + MLflow
                            └─ src/simulator.py     100k tournament runs → simulation_results.csv
                                 └─ app/dashboard.py    Streamlit UI (reads processed/ only)
```

`src/pipeline.py` orchestrates all seven stages; each module can also be run independently.

---

## Project structure

```
.
├── src/
│   ├── data_loader.py      # load/filter results, shootouts, fixtures
│   ├── elo_calculator.py   # Elo ratings from scratch + rank derivation
│   ├── feature_builder.py  # 18 leak-free pre-match features
│   ├── poisson_model.py    # Stage 1: Poisson GLM
│   ├── xgboost_model.py    # Stage 2: XGBoost + Optuna + SHAP + MLflow
│   ├── simulator.py        # Stage 3: Monte Carlo tournament
│   └── pipeline.py         # end-to-end orchestrator
├── app/
│   └── dashboard.py        # Streamlit 4-view dashboard
├── scripts/
│   ├── generate_fixtures.py  # builds wc2026_fixtures.csv
│   └── smoke_test.py         # full pipeline on synthetic data
├── tests/                  # pytest suite (48 tests)
├── data/
│   ├── raw/                # input CSVs (only wc2026_fixtures.csv is committed)
│   └── processed/          # pipeline outputs (gitignored)
├── docs/superpowers/       # design spec + implementation plan
└── requirements.txt
```

---

## Setup

Requires Python 3.11+ (developed on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins `setuptools<81` on purpose — `shap`/`numba` still import
> `pkg_resources`, which was removed in setuptools 81+.

---

## Data

The model trains on the public Kaggle dataset
**[martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)**.
These files are **not committed** — download them and place them in `data/raw/`:

| File | Purpose |
|------|---------|
| `results.csv` | Match results 1872–2024 (filtered to 1990+ for training) |
| `goalscorers.csv` | Scorer-level data (reserved for future enrichment) |
| `shootouts.csv` | Penalty shootout winners (knockout tiebreaker) |
| `wc2026_fixtures.csv` | **Already committed** — remaining WC 2026 fixtures (regenerate with `python scripts/generate_fixtures.py`) |

**Train / validation split**
- Train: matches with `date <= 2018-12-31`
- Validation (holdout): `tournament == 'FIFA World Cup'` and `2019-01-01 <= date <= 2022-12-31` (WC 2022)
- Completed WC 2026 matches are ignored; only future fixtures are simulated.

---

## Usage

**Run the full pipeline** (after placing the Kaggle CSVs in `data/raw/`):

```bash
python -m src.pipeline --trials 100 --simulations 100000
```

This writes `elo_history.csv`, `features.csv`, model `.pkl` files, `shap_values.csv`, and
`simulation_results.csv` into `data/processed/`, logging every model + metric to MLflow.
Timestamped progress (per-stage `[i/6]` markers and `~5%` increment updates within the long
loops) is logged to the console throughout.

### Checkpoints, progress, and resume

The pipeline runs as six checkpointed stages, each writing one artifact to `data/processed/`:

| # | Stage | Artifact |
|---|-------|----------|
| 1 | `elo` | `elo_history.csv` |
| 2 | `features` | `features.csv` (+ `poisson_model.pkl`) |
| 3 | `xgb_3class` | `xgb_3class.pkl` |
| 4 | `xgb_binary` | `xgb_binary.pkl` |
| 5 | `shap` | `shap_values.csv` |
| 6 | `simulate` | `simulation_results.csv` |

**Check where the pipeline is** (which stages are done, what runs next):

```bash
python -m src.pipeline --status
```

**Resume after a failure.** A plain run **automatically skips stages whose artifact already
exists** and starts at the first incomplete one — so if it crashes in stage 5, just re-run and
it picks up at stage 5 instead of retraining from scratch:

```bash
python -m src.pipeline                      # resume from first missing artifact
```

**Force or target specific stages:**

```bash
python -m src.pipeline --force              # ignore checkpoints, run all 6 from scratch
python -m src.pipeline --start-from xgb_3class   # re-run from a chosen stage onward
```

> **Performance:** the Monte Carlo simulator scores every distinct matchup **once** (~2,256
> ordered pairs among 48 teams) and caches the probabilities, so the 100k-tournament loop does
> zero model calls — a full 100k run completes in ~25 seconds. Lower `--simulations 10000` for
> even faster, still-stable estimates while iterating.

**Launch the dashboard:**

```bash
streamlit run app/dashboard.py        # http://localhost:8501
```

Four views: Match Predictor · Champion Probabilities · Group Stage Standings · SHAP Feature Importance.

**Smoke test** (no Kaggle data needed — runs every stage on synthetic data):

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

**Tests:**

```bash
python -m pytest tests/ -q            # 51 tests
```

---

## Key design choices

- **Elo computed from scratch** (not eloratings.net): all teams start at 1500 on 1990-01-01;
  K = 40 (World Cup) / 30 (continental) / 20 (other). FIFA rank is derived by ordering active
  teams (≥1 match in the prior 12 months) by Elo.
- **18 pre-match features**, all computed with `groupby + shift(1)` / expanding windows so no
  future information leaks. `squad_value_ratio` is hardcoded `0.0` (TODO: Transfermarkt data).
- **Probability calibration:** `CalibratedClassifierCV(method='sigmoid')` guarantees outputs sum to 1.
- **Temporal validation only:** `TimeSeriesSplit` — never a random shuffle.
- **Knockout stage** suppresses the draw class (binary model); ties broken by historical penalty
  shootout win rate, falling back to 50/50.

## WC 2022 holdout targets

| Metric | Target |
|--------|--------|
| 3-class accuracy | > 0.55 |
| Log-loss | < 0.95 |
| Brier score | < 0.22 |
| Ranked Probability Score | < 0.20 |

Inspect actual metrics after a run:

```bash
python -c "import mlflow; c=mlflow.tracking.MlflowClient(); e=c.get_experiment_by_name('wc2026'); \
[print(r.data.tags.get('mlflow.runName'), r.data.metrics) for r in c.search_runs(e.experiment_id, order_by=['start_time DESC'])[:2]]"
```

---

## Tech stack

Python 3.11+ · pandas · numpy · statsmodels · scipy · xgboost · optuna · shap ·
scikit-learn · streamlit · plotly · mlflow · pytest
