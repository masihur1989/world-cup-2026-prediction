import pandas as pd
import numpy as np
import pytest
from src.poisson_model import PoissonGoalModel, add_poisson_features

@pytest.fixture
def small_feature_df():
    """Minimal feature DataFrame enough to fit the Poisson model."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "date":             pd.date_range("1995-01-01", periods=n, freq="7D"),
        "team_a":           ["Germany"] * n,
        "team_b":           ["France"] * n,
        "tournament":       ["FIFA World Cup"] * n,
        "elo_diff":         rng.normal(0, 100, n),
        "fifa_rank_diff":   rng.integers(-20, 20, n).astype(float),
        "form_A":           rng.uniform(0, 30, n),
        "form_B":           rng.uniform(0, 30, n),
        "goals_scored_avg_A": rng.uniform(0.5, 3.0, n),
        "goals_scored_avg_B": rng.uniform(0.5, 3.0, n),
        "goals_conceded_avg_A": rng.uniform(0.5, 2.0, n),
        "goals_conceded_avg_B": rng.uniform(0.5, 2.0, n),
        "h2h_win_rate_A":   rng.uniform(0, 1, n),
        "h2h_goal_diff":    rng.normal(0, 1, n),
        "squad_value_ratio": np.zeros(n),
        "stage_weight":     np.ones(n, dtype=int),
        "rest_days_diff":   rng.integers(-14, 14, n).astype(float),
        "penalty_win_rate_A": rng.uniform(0, 1, n),
        "outcome":          rng.choice([1, 0, -1], n),
        **{f"conf_A_{c}": np.zeros(n, dtype=int) for c in ["UEFA","CONMEBOL","CAF","AFC","CONCACAF","OFC"]},
        **{f"conf_B_{c}": np.zeros(n, dtype=int) for c in ["UEFA","CONMEBOL","CAF","AFC","CONCACAF","OFC"]},
    })

def test_poisson_model_fits(small_feature_df):
    model = PoissonGoalModel()
    model.fit(small_feature_df)
    assert model.model_a is not None
    assert model.model_b is not None

def test_predict_lambda_returns_positive(small_feature_df):
    model = PoissonGoalModel()
    model.fit(small_feature_df)
    row = small_feature_df.iloc[0]
    la, lb = model.predict_lambda(row)
    assert la > 0
    assert lb > 0

def test_simulate_match_probabilities_sum_to_1(small_feature_df):
    model = PoissonGoalModel()
    model.fit(small_feature_df)
    row = small_feature_df.iloc[0]
    la, lb = model.predict_lambda(row)
    p_win, p_draw, p_loss = model.simulate_match(la, lb, n=10_000)
    assert abs(p_win + p_draw + p_loss - 1.0) < 1e-6

def test_simulate_probabilities_between_0_and_1(small_feature_df):
    model = PoissonGoalModel()
    model.fit(small_feature_df)
    row = small_feature_df.iloc[0]
    la, lb = model.predict_lambda(row)
    p_win, p_draw, p_loss = model.simulate_match(la, lb, n=10_000)
    for p in [p_win, p_draw, p_loss]:
        assert 0.0 <= p <= 1.0

def test_add_poisson_features_adds_two_columns(small_feature_df):
    model = PoissonGoalModel()
    model.fit(small_feature_df)
    result = add_poisson_features(small_feature_df, model)
    assert "lambda_a" in result.columns
    assert "p_win_poisson" in result.columns

def test_poisson_save_load(tmp_path, small_feature_df):
    model = PoissonGoalModel()
    model.fit(small_feature_df)
    path = str(tmp_path / "poisson.pkl")
    model.save(path)
    model2 = PoissonGoalModel()
    model2.load(path)
    row = small_feature_df.iloc[0]
    la1, lb1 = model.predict_lambda(row)
    la2, lb2 = model2.predict_lambda(row)
    assert abs(la1 - la2) < 1e-6
    assert abs(lb1 - lb2) < 1e-6
