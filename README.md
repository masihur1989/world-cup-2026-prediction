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

> 📘 **Full guide:** see [docs/USAGE.md](docs/USAGE.md) for what's built, how to run every feature, and how to update the data round-by-round.
> 🧠 **ML build-out:** see [docs/MODEL.md](docs/MODEL.md) for the modeling details — Elo, features, the three stages, calibration, and evaluation.

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
├── tests/                  # pytest suite (74 tests)
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
pip install -r requirements-train.txt   # full stack: pipeline, training, tests
```

`requirements.txt` holds **only the dashboard runtime** (pandas, numpy, streamlit, plotly) so the
deployed app builds fast; `requirements-train.txt` pulls that in plus the training stack
(xgboost, statsmodels, optuna, shap, scikit-learn, mlflow, pytest). To run *only* the dashboard,
`pip install -r requirements.txt` is enough.

> **Note:** `requirements-train.txt` pins `setuptools<81` — `shap`/`numba` still import
> `pkg_resources`, removed in setuptools 81+.

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

### Fixture-driven prediction & dated snapshots

The simulator is driven entirely by `data/raw/wc2026_fixtures.csv` — real groups, the
official Round-of-32 slot pairings, and standard-adjacency R16→Final. Each run is saved as a
**dated snapshot** under `data/processed/predictions/` (`{date}__{label}.csv` plus a
`__bracket.csv`), indexed in `predictions/index.csv`.

**Locked pre-tournament baseline** (committed): `2026-06-10__pre_tournament`, simulated with no
real results.

**Round-by-round updates.** As each round finishes, fill in that round's outcomes in
`data/raw/wc2026_actual_results.csv` (`stage,team_a,team_b,score_a,score_b,winner`). Those
results are pinned (not re-simulated), and a new snapshot is written for the remaining matches —
earlier snapshots stay immutable:

```bash
python -m src.pipeline --start-from simulate --as-of 2026-06-28 --label after_groups
```

`--as-of DATE` sets the snapshot date and the as-of date for matchup features; `--label TEXT`
names it. Re-running an existing `date/label` requires `--force`.

**Launch the dashboard:**

```bash
streamlit run app/dashboard.py        # http://localhost:8501
```

Four views: Match Predictor · Champion Probabilities · Group Stage Standings (real advancement
odds) · **Tournament Bracket** (probabilistic — each match shows the teams most likely to reach
it). A sidebar selector switches between saved snapshots.

### Deploying the dashboard

The dashboard is stateless and read-only (no model code at runtime), and the committed
`2026-06-10__pre_tournament` snapshot is self-sufficient — a fresh clone runs as-is.

**Streamlit Community Cloud (recommended, free):**
1. Push the repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io): New app → your repo, branch `master`,
   main file `app/dashboard.py`, Python 3.12 → Deploy. It installs the slim `requirements.txt`.

**Container host (Render / Fly.io / Cloud Run)** — for a custom domain or always-on:

```bash
docker build -t wc2026-dashboard .
docker run -p 8501:8501 wc2026-dashboard      # http://localhost:8501
```

The `Dockerfile` installs only the dashboard runtime and copies just what the app reads
(the app, `src/fixtures_bracket.py`, the fixtures CSV, and the committed snapshots). It binds to
`$PORT`, so Render/Fly/Cloud Run work out of the box.

**Publishing updated predictions:** generate a new snapshot, commit it, and push — the deployment
redeploys (Streamlit Cloud) or rebuild the image (container):

```bash
python -m src.pipeline --start-from simulate --as-of 2026-06-28 --label after_groups
git add -f data/processed/predictions/2026-06-28__after_groups*.csv data/processed/predictions/index.csv
git commit -m "snapshot: after group stage" && git push
```

**Smoke test** (no Kaggle data needed — runs every stage on synthetic data):

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

**Tests:**

```bash
python -m pytest tests/ -q            # 74 tests
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
