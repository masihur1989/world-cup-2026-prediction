import pandas as pd
import numpy as np
import pytest
from src.simulator import TournamentSimulator, WC2026_GROUPS, get_8_best_third

@pytest.fixture
def mock_model():
    """Stub XGBoost model that always returns equal probabilities."""
    class MockModel:
        def predict_proba(self, X):
            n = len(X)
            return np.full((n, 3), 1/3)
        label_encoder = type("LE", (), {"classes_": np.array([-1, 0, 1])})()
    return MockModel()

@pytest.fixture
def mock_features():
    """Minimal feature DataFrame with all required columns."""
    from src.xgboost_model import FEATURE_COLS
    rng = np.random.default_rng(1)
    teams = [t for group in WC2026_GROUPS.values() for t in group]
    rows = []
    for t_a in teams[:4]:
        for t_b in teams[4:8]:
            row = {col: rng.standard_normal() for col in FEATURE_COLS}
            row["team_a"] = t_a
            row["team_b"] = t_b
            rows.append(row)
    return pd.DataFrame(rows)

def test_wc2026_groups_has_12_groups():
    assert len(WC2026_GROUPS) == 12

def test_wc2026_groups_has_4_teams_each():
    for group, teams in WC2026_GROUPS.items():
        assert len(teams) == 4, f"Group {group} has {len(teams)} teams"

def test_wc2026_total_48_unique_teams():
    all_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
    assert len(all_teams) == 48
    assert len(set(all_teams)) == 48

def test_get_8_best_third():
    third_place = [
        {"team": f"Team{i}", "pts": i % 5, "gd": i % 3, "gf": i % 7}
        for i in range(12)
    ]
    best = get_8_best_third(third_place)
    assert len(best) == 8

def test_simulate_group_returns_standings(mock_model, mock_features):
    from src.feature_builder import CONFEDERATION
    penalty_rates = {t: 0.5 for group in WC2026_GROUPS.values() for t in group}
    sim = TournamentSimulator(mock_model, penalty_rates, mock_features)
    standings = sim.simulate_group("A")
    assert len(standings) == 4
    for entry in standings:
        assert "team" in entry
        assert "pts" in entry
        assert "gd" in entry
        assert "gf" in entry

def test_simulate_knockout_returns_string(mock_model, mock_features):
    penalty_rates = {t: 0.5 for group in WC2026_GROUPS.values() for t in group}
    sim = TournamentSimulator(mock_model, penalty_rates, mock_features)
    winner = sim.simulate_knockout_match("Germany", "France")
    assert winner in {"Germany", "France"}

def test_run_returns_dataframe_with_48_rows(mock_model, mock_features):
    penalty_rates = {t: 0.5 for group in WC2026_GROUPS.values() for t in group}
    sim = TournamentSimulator(mock_model, penalty_rates, mock_features)
    result = sim.run(n_simulations=10)
    assert len(result) == 48
    assert "p_champion" in result.columns
    assert "p_finalist" in result.columns
    assert "p_semifinalist" in result.columns

def test_champion_probabilities_sum_to_1(mock_model, mock_features):
    penalty_rates = {t: 0.5 for group in WC2026_GROUPS.values() for t in group}
    sim = TournamentSimulator(mock_model, penalty_rates, mock_features)
    result = sim.run(n_simulations=100)
    assert abs(result["p_champion"].sum() - 1.0) < 0.01
