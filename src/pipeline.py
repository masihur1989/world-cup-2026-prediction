"""
Full pipeline orchestrator with logging, progress, and checkpoint-based resume.

Run the whole thing (resumes automatically from the first incomplete stage):
    python -m src.pipeline

Inspect progress / which stages are done:
    python -m src.pipeline --status

Force a clean run from scratch, or restart at a specific stage:
    python -m src.pipeline --force
    python -m src.pipeline --start-from xgb_3class

Each stage writes a checkpoint artifact to data/processed/. On a normal run,
stages whose artifact already exists are skipped, so a failure part-way through
can be resumed without repeating completed work.
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_results, load_shootouts, load_fixtures
from src.elo_calculator import compute_elo_history, save_elo_history
from src.feature_builder import (
    build_features,
    save_features,
    compute_penalty_win_rates,
)
from src.poisson_model import PoissonGoalModel, add_poisson_features
from src.xgboost_model import WorldCupXGBModel, FEATURE_COLS, train_and_log
from src.simulator import TournamentSimulator, save_simulation_results

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
Path("mlruns").mkdir(exist_ok=True)

# Ordered stages and the artifact each one produces (its checkpoint).
STAGE_ORDER = ["elo", "features", "xgb_3class", "xgb_binary", "shap", "simulate"]
ARTIFACTS = {
    "elo":        PROCESSED / "elo_history.csv",
    "features":   PROCESSED / "features.csv",
    "xgb_3class": PROCESSED / "xgb_3class.pkl",
    "xgb_binary": PROCESSED / "xgb_binary.pkl",
    "shap":       PROCESSED / "shap_values.csv",
    "simulate":   PROCESSED / "simulation_results.csv",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def _progress(label: str):
    """Return a callback that logs percentage completion for a long loop."""
    def cb(done: int, total: int):
        log.info("  %s: %d/%d (%.0f%%)", label, done, total, 100 * done / total)
    return cb


# ── Lazy getters: pull inputs from memory (ctx) or load from disk artifacts ──

def _get_results(ctx):
    if "results" not in ctx:
        ctx["results"] = load_results()
    return ctx["results"]


def _get_shootouts(ctx):
    if "shootouts" not in ctx:
        ctx["shootouts"] = load_shootouts()
    return ctx["shootouts"]


def _get_fixtures(ctx):
    if "fixtures" not in ctx:
        ctx["fixtures"] = load_fixtures()
    return ctx["fixtures"]


def _get_elo(ctx):
    if "elo" not in ctx:
        ctx["elo"] = pd.read_csv(ARTIFACTS["elo"], parse_dates=["date"])
    return ctx["elo"]


def _get_features(ctx):
    if "features" not in ctx:
        ctx["features"] = pd.read_csv(ARTIFACTS["features"], parse_dates=["date"])
    return ctx["features"]


def _get_model_3class(ctx):
    if "model_3class" not in ctx:
        m = WorldCupXGBModel(mode="3class")
        m.load(str(ARTIFACTS["xgb_3class"]))
        ctx["model_3class"] = m
    return ctx["model_3class"]


# ── Stage implementations ────────────────────────────────────────────────────

def stage_elo(ctx, cfg):
    results = _get_results(ctx)
    elo = compute_elo_history(results)
    save_elo_history(elo)
    ctx["elo"] = elo
    log.info("  elo_history: %d snapshots", len(elo))


def stage_features(ctx, cfg):
    results = _get_results(ctx)
    shootouts = _get_shootouts(ctx)
    fixtures = _get_fixtures(ctx)
    elo = _get_elo(ctx)

    features = build_features(
        results, elo, shootouts, fixtures=fixtures,
        progress_callback=_progress("base features"),
    )
    log.info("  base features: %d rows × %d cols", len(features), len(features.columns))

    # Poisson features (17-18). Train on <=2018; score only non-train rows.
    train_mask = features["date"] <= "2018-12-31"
    poisson = PoissonGoalModel()
    poisson.fit(features[train_mask])
    log.info("  poisson GLM fitted; scoring %d non-train rows", int((~train_mask).sum()))
    non_train = add_poisson_features(
        features[~train_mask].copy(), poisson,
        progress_callback=_progress("poisson features"),
    )
    features = pd.concat([
        features[train_mask].assign(lambda_a=1.5, p_win_poisson=0.45),
        non_train,
    ]).sort_values("date").reset_index(drop=True)

    save_features(features)
    poisson.save(str(PROCESSED / "poisson_model.pkl"))
    ctx["features"] = features


def stage_xgb_3class(ctx, cfg):
    features = _get_features(ctx)
    model = train_and_log(features, mode="3class", n_trials=cfg["trials"])
    model.save(str(ARTIFACTS["xgb_3class"]))
    ctx["model_3class"] = model


def stage_xgb_binary(ctx, cfg):
    features = _get_features(ctx)
    model = train_and_log(features, mode="binary", n_trials=cfg["trials"])
    model.save(str(ARTIFACTS["xgb_binary"]))


def stage_shap(ctx, cfg):
    features = _get_features(ctx)
    model = _get_model_3class(ctx)
    val = features[
        (features["tournament"] == "FIFA World Cup") &
        (features["date"] >= "2019-01-01") &
        (features["date"] <= "2022-12-31")
    ].dropna(subset=["outcome"])
    if len(val) == 0:
        log.warning("  no validation rows; writing empty SHAP file")
        pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": 0.0}).to_csv(
            ARTIFACTS["shap"], index=False
        )
        return

    X_val = val[FEATURE_COLS].fillna(0.0)
    shap_vals = model.compute_shap(X_val)
    # SHAP output shape varies by version; collapse every axis except the one
    # whose length matches the feature count.
    arr = np.abs(np.array(shap_vals))
    n_features = len(FEATURE_COLS)
    feature_axes = [ax for ax, size in enumerate(arr.shape) if size == n_features]
    if not feature_axes:
        raise ValueError(
            f"No SHAP axis matches feature count {n_features}; got shape {arr.shape}"
        )
    feature_axis = feature_axes[-1]
    other_axes = tuple(ax for ax in range(arr.ndim) if ax != feature_axis)
    mean_shap = arr.mean(axis=other_axes)
    pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_shap}).to_csv(
        ARTIFACTS["shap"], index=False
    )


def stage_simulate(ctx, cfg):
    features = _get_features(ctx)
    shootouts = _get_shootouts(ctx)
    model = _get_model_3class(ctx)
    penalty_rates = compute_penalty_win_rates(shootouts)
    sim = TournamentSimulator(model, penalty_rates, features)
    log.info("  precomputing matchup probabilities...")
    sim.precompute_probabilities()
    results = sim.run(
        n_simulations=cfg["simulations"],
        progress_callback=_progress("simulation"),
    )
    save_simulation_results(results)
    ctx["sim_results"] = results
    log.info("  top 5 champions:\n%s",
             results[["team", "p_champion", "confederation"]].head().to_string(index=False))


STAGE_FUNCS = {
    "elo": stage_elo,
    "features": stage_features,
    "xgb_3class": stage_xgb_3class,
    "xgb_binary": stage_xgb_binary,
    "shap": stage_shap,
    "simulate": stage_simulate,
}


# ── Orchestration ────────────────────────────────────────────────────────────

def first_incomplete_stage() -> int:
    """Index of the first stage whose checkpoint artifact is missing."""
    for i, stage in enumerate(STAGE_ORDER):
        if not ARTIFACTS[stage].exists():
            return i
    return len(STAGE_ORDER)


def print_status() -> None:
    n = len(STAGE_ORDER)
    print("Pipeline checkpoint status:")
    for i, stage in enumerate(STAGE_ORDER):
        path = ARTIFACTS[stage]
        if path.exists():
            ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  [✓] {i+1}/{n}  {stage:<11}  {path.name:<24}  {ts}")
        else:
            print(f"  [ ] {i+1}/{n}  {stage:<11}  {path.name:<24}  (missing)")
    nxt = first_incomplete_stage()
    if nxt == n:
        print("\nAll stages complete. Use --force or --start-from to re-run.")
    else:
        print(f"\nNext stage to run: {STAGE_ORDER[nxt]} "
              f"(resume with: python -m src.pipeline)")


def run_pipeline(trials=100, simulations=100_000, start_from=None, force=False):
    cfg = {"trials": trials, "simulations": simulations}
    n = len(STAGE_ORDER)

    if force:
        start_idx = 0
    elif start_from:
        start_idx = STAGE_ORDER.index(start_from)
    else:
        start_idx = first_incomplete_stage()

    if start_idx >= n:
        log.info("All stages already complete. Use --force or --start-from to re-run.")
        return

    log.info("=" * 60)
    log.info("WC 2026 Prediction Pipeline  (trials=%d, simulations=%d)", trials, simulations)
    log.info("Starting from stage '%s' [%d/%d]", STAGE_ORDER[start_idx], start_idx + 1, n)
    log.info("=" * 60)

    ctx: dict = {}
    t0 = time.time()
    for i, stage in enumerate(STAGE_ORDER):
        if i < start_idx:
            log.info("[%d/%d] %-11s SKIP (checkpoint: %s)", i + 1, n, stage, ARTIFACTS[stage].name)
            continue
        log.info("[%d/%d] %-11s started", i + 1, n, stage)
        t = time.time()
        STAGE_FUNCS[stage](ctx, cfg)
        log.info("[%d/%d] %-11s done in %.1fs → %s",
                 i + 1, n, stage, time.time() - t, ARTIFACTS[stage].name)

    log.info("=" * 60)
    log.info("Pipeline complete in %.1f minutes.", (time.time() - t0) / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WC 2026 prediction pipeline")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials per XGBoost model")
    parser.add_argument("--simulations", type=int, default=100_000, help="Monte Carlo iterations")
    parser.add_argument("--start-from", choices=STAGE_ORDER, default=None,
                        help="Restart at this stage, ignoring its checkpoint")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages from scratch, ignoring checkpoints")
    parser.add_argument("--status", action="store_true",
                        help="Show checkpoint status and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        run_pipeline(
            trials=args.trials,
            simulations=args.simulations,
            start_from=args.start_from,
            force=args.force,
        )
