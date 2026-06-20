"""
Full pipeline orchestrator. Run: python -m src.pipeline
Each stage can also be run independently by calling its public functions.
"""
import time
from pathlib import Path

import pandas as pd
import mlflow

from src.data_loader import load_results, load_shootouts, load_fixtures
from src.elo_calculator import compute_elo_history, save_elo_history
from src.feature_builder import build_features, save_features
from src.poisson_model import PoissonGoalModel, add_poisson_features
from src.xgboost_model import WorldCupXGBModel, FEATURE_COLS, train_and_log
from src.simulator import TournamentSimulator, WC2026_GROUPS, save_simulation_results
from src.feature_builder import CONFEDERATION

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("mlruns").mkdir(exist_ok=True)


def run_pipeline(
    optuna_trials: int = 100,
    n_simulations: int = 100_000,
) -> None:
    t0 = time.time()
    print("=" * 60)
    print("WC 2026 Prediction Pipeline")
    print("=" * 60)

    # ── Stage 0: Load data ──────────────────────────────────────────
    print("\n[1/7] Loading data...")
    results  = load_results()
    shootouts = load_shootouts()
    fixtures  = load_fixtures()
    print(f"  results:   {len(results):,} rows")
    print(f"  shootouts: {len(shootouts):,} rows")
    print(f"  fixtures:  {len(fixtures):,} rows")

    # ── Stage 1: Elo ────────────────────────────────────────────────
    print("\n[2/7] Computing Elo history...")
    elo_history = compute_elo_history(results)
    save_elo_history(elo_history)
    print(f"  elo_history: {len(elo_history):,} snapshots → data/processed/elo_history.csv")

    # ── Stage 2: Base features (16) ─────────────────────────────────
    print("\n[3/7] Building base features (1-16)...")
    features = build_features(results, elo_history, shootouts, fixtures=fixtures)
    print(f"  features: {len(features):,} rows × {len(features.columns)} cols")

    # ── Stage 3: Poisson GLM (features 17-18) ───────────────────────
    print("\n[4/7] Fitting Poisson GLM and adding features 17-18...")
    train_mask = features["date"] <= "2018-12-31"
    poisson = PoissonGoalModel()
    poisson.fit(features[train_mask])

    # Add Poisson features only to validation + fixture rows (skip full corpus for speed)
    non_train = features[~train_mask].copy()
    non_train = add_poisson_features(non_train, poisson)
    features = pd.concat([
        features[train_mask].assign(lambda_a=1.5, p_win_poisson=0.45),  # train rows get defaults
        non_train,
    ]).sort_values("date").reset_index(drop=True)

    save_features(features)
    print(f"  features saved → data/processed/features.csv")

    poisson.save("data/processed/poisson_model.pkl")

    # ── Stage 4: XGBoost 3-class (group stage) ──────────────────────
    print(f"\n[5/7] Training XGBoost 3-class model ({optuna_trials} Optuna trials)...")
    model_3class = train_and_log(features, mode="3class", n_trials=optuna_trials)
    model_3class.save("data/processed/xgb_3class.pkl")
    print("  model saved → data/processed/xgb_3class.pkl")

    # ── Stage 5: XGBoost binary (knockout) ──────────────────────────
    print(f"\n[6/7] Training XGBoost binary model ({optuna_trials} Optuna trials)...")
    model_binary = train_and_log(features, mode="binary", n_trials=optuna_trials)
    model_binary.save("data/processed/xgb_binary.pkl")
    print("  model saved → data/processed/xgb_binary.pkl")

    # ── Stage 6: SHAP values ────────────────────────────────────────
    val_features = features[
        (features["tournament"] == "FIFA World Cup") &
        (features["date"] >= "2019-01-01") &
        (features["date"] <= "2022-12-31")
    ].dropna(subset=["outcome"])
    if len(val_features) > 0:
        X_val = val_features[FEATURE_COLS].fillna(0.0)
        shap_vals = model_3class.compute_shap(X_val)
        # Save mean absolute SHAP per feature
        import numpy as np
        if isinstance(shap_vals, list):
            mean_shap = np.abs(np.array(shap_vals)).mean(axis=(0, 1))
        else:
            arr = np.abs(shap_vals)
            mean_shap = arr.mean(axis=tuple(range(arr.ndim - 1))) if arr.ndim > 2 else arr.mean(axis=0)
        shap_df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_shap})
        shap_df.to_csv("data/processed/shap_values.csv", index=False)
        print("  shap_values saved → data/processed/shap_values.csv")

    # ── Stage 7: Monte Carlo simulation ─────────────────────────────
    print(f"\n[7/7] Running Monte Carlo simulation ({n_simulations:,} iterations)...")
    from src.feature_builder import compute_penalty_win_rates
    penalty_rates = compute_penalty_win_rates(shootouts)
    sim = TournamentSimulator(model_3class, penalty_rates, features)
    sim_results = sim.run(n_simulations=n_simulations)
    save_simulation_results(sim_results)
    print("  simulation_results saved → data/processed/simulation_results.csv")

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed/60:.1f} minutes.")
    print("Top 5 championship probabilities:")
    print(sim_results[["team", "p_champion", "confederation"]].head())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=100_000)
    args = parser.parse_args()
    run_pipeline(optuna_trials=args.trials, n_simulations=args.simulations)
