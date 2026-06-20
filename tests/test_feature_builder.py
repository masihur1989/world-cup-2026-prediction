import pandas as pd
import numpy as np
import pytest
from src.feature_builder import (
    build_team_perspective,
    compute_penalty_win_rates,
    build_features,
    CONFEDERATION,
    STAGE_WEIGHT,
)

def test_confederation_covers_48_teams():
    assert len(CONFEDERATION) >= 48

def test_confederation_all_valid_values():
    valid = {"UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"}
    assert set(CONFEDERATION.values()).issubset(valid)

def test_stage_weight_values():
    assert STAGE_WEIGHT["Group"] == 1
    assert STAGE_WEIGHT["Round of 32"] == 2
    assert STAGE_WEIGHT["Quarterfinal"] == 3
    assert STAGE_WEIGHT["Semifinal"] == 4
    assert STAGE_WEIGHT["Final"] == 5

def test_build_team_perspective_shape(mini_results):
    tp = build_team_perspective(mini_results)
    assert len(tp) == 2 * len(mini_results)
    required = {"date", "team", "opponent", "gf", "ga", "win", "draw", "loss", "points"}
    assert required.issubset(tp.columns)

def test_team_perspective_no_future_leakage(mini_results):
    tp = build_team_perspective(mini_results)
    germany = tp[tp["team"] == "Germany"].sort_values("date")
    assert pd.isna(germany.iloc[0]["form"])

def test_compute_penalty_win_rates(mini_shootouts):
    rates = compute_penalty_win_rates(mini_shootouts)
    assert isinstance(rates, dict)
    assert rates["Germany"] == 1.0
    assert rates["France"] == 0.5

def test_build_features_has_16_base_columns(mini_results, mini_shootouts):
    from src.elo_calculator import compute_elo_history
    elo_history = compute_elo_history(mini_results)
    features = build_features(mini_results, elo_history, mini_shootouts)
    base_feature_cols = [
        "elo_diff", "fifa_rank_diff", "form_A", "form_B",
        "goals_scored_avg_A", "goals_scored_avg_B",
        "goals_conceded_avg_A", "goals_conceded_avg_B",
        "h2h_win_rate_A", "h2h_goal_diff",
        "squad_value_ratio", "stage_weight", "rest_days_diff",
        "conf_A_UEFA", "conf_A_CONMEBOL", "conf_A_CAF",
        "conf_A_AFC", "conf_A_CONCACAF", "conf_A_OFC",
        "conf_B_UEFA", "conf_B_CONMEBOL", "conf_B_CAF",
        "conf_B_AFC", "conf_B_CONCACAF", "conf_B_OFC",
        "penalty_win_rate_A",
    ]
    for col in base_feature_cols:
        assert col in features.columns, f"Missing column: {col}"

def test_squad_value_ratio_always_zero(mini_results, mini_shootouts):
    from src.elo_calculator import compute_elo_history
    elo_history = compute_elo_history(mini_results)
    features = build_features(mini_results, elo_history, mini_shootouts)
    assert (features["squad_value_ratio"] == 0.0).all()

def test_features_no_future_rows_used(mini_results, mini_shootouts):
    from src.elo_calculator import compute_elo_history
    elo_history = compute_elo_history(mini_results)
    features = build_features(mini_results, elo_history, mini_shootouts)
    earliest = features.sort_values("date").iloc[0]
    assert pd.isna(earliest["form_A"]) or earliest["form_A"] == 0.0

def test_build_features_includes_outcome(mini_results, mini_shootouts):
    from src.elo_calculator import compute_elo_history
    elo_history = compute_elo_history(mini_results)
    features = build_features(mini_results, elo_history, mini_shootouts)
    assert "outcome" in features.columns
    assert set(features["outcome"].dropna().unique()).issubset({1, 0, -1})
