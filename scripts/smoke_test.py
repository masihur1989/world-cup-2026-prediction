import pandas as pd
import numpy as np
from src.data_loader import load_results
from src.elo_calculator import compute_elo_history
from src.feature_builder import build_features, compute_penalty_win_rates
from src.poisson_model import PoissonGoalModel, add_poisson_features
from src.xgboost_model import WorldCupXGBModel, FEATURE_COLS
from src.simulator import TournamentSimulator, WC2026_GROUPS

# Generate synthetic results.csv with 500 rows (all 48 WC2026 teams)
rng = np.random.default_rng(42)
all_teams = [t for group in WC2026_GROUPS.values() for t in group]
n = 500
teams_a = rng.choice(all_teams, n)
teams_b = rng.choice(all_teams, n)
results = pd.DataFrame({
    "date": pd.date_range("1995-01-01", periods=n, freq="14D"),
    "home_team": teams_a,
    "away_team": teams_b,
    "home_score": rng.integers(0, 5, n),
    "away_score": rng.integers(0, 4, n),
    "tournament": rng.choice(
        ["FIFA World Cup", "Friendly", "UEFA Euro", "Copa America"], n
    ),
    "city": ["Test"] * n,
    "country": ["Test"] * n,
    "neutral": [False] * n,
})
# Filter self-matches
results = results[results["home_team"] != results["away_team"]].head(400)

so_home = rng.choice(all_teams, 20)
so_away = rng.choice(all_teams, 20)
shootouts = pd.DataFrame({
    "date": pd.date_range("1995-01-01", periods=20, freq="60D"),
    "home_team": so_home,
    "away_team": so_away,
    # winner is always one of the two participants (matches real shootouts.csv)
    "winner": [h if b else a for h, a, b in zip(so_home, so_away, rng.integers(0, 2, 20))],
})

elo_history = compute_elo_history(results)
print(f"Elo history: {len(elo_history)} rows")

features = build_features(results, elo_history, shootouts)
print(f"Features: {len(features)} rows")

poisson = PoissonGoalModel()
poisson.fit(features.dropna(subset=["home_score"]))
features = features.assign(lambda_a=1.5, p_win_poisson=0.45)
print("Poisson model fitted")

X = features[FEATURE_COLS].fillna(0.0)
y = features["outcome"].fillna(1).astype(int).values

model_3class = WorldCupXGBModel(mode="3class")
params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 30,
          "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1}
model_3class.fit(X, y, features["date"], params)
print("XGBoost 3-class fitted")

penalty_rates = compute_penalty_win_rates(shootouts)
sim = TournamentSimulator(model_3class, penalty_rates, features)
sim_results = sim.run(n_simulations=100)
print("Monte Carlo simulation OK")
print(sim_results[["team", "p_champion"]].head())
print("\nSmoke test PASSED")
