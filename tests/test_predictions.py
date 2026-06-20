import pandas as pd
import pytest
from src.predictions import (
    save_prediction_snapshot, load_actual_results, PREDICTIONS_DIR,
)


@pytest.fixture
def team_df():
    return pd.DataFrame({"team": ["Brazil", "France"],
                         "p_champion": [0.6, 0.4], "confederation": ["CONMEBOL", "UEFA"]})


@pytest.fixture
def bracket_df():
    return pd.DataFrame({"round": ["Final"], "match_index": [0],
                         "team": ["Brazil"], "p_reach": [0.6]})


def test_save_snapshot_writes_files(tmp_path, monkeypatch, team_df, bracket_df):
    monkeypatch.setattr("src.predictions.PROCESSED", tmp_path)
    monkeypatch.setattr("src.predictions.PREDICTIONS_DIR", tmp_path / "predictions")
    paths = save_prediction_snapshot(team_df, bracket_df, "2026-06-10",
                                     "pre_tournament", n_simulations=100)
    assert (tmp_path / "predictions" / "2026-06-10__pre_tournament.csv").exists()
    assert (tmp_path / "predictions" / "2026-06-10__pre_tournament__bracket.csv").exists()
    assert (tmp_path / "predictions" / "index.csv").exists()
    assert (tmp_path / "simulation_results.csv").exists()
    assert (tmp_path / "bracket.csv").exists()


def test_save_snapshot_no_overwrite(tmp_path, monkeypatch, team_df, bracket_df):
    monkeypatch.setattr("src.predictions.PROCESSED", tmp_path)
    monkeypatch.setattr("src.predictions.PREDICTIONS_DIR", tmp_path / "predictions")
    save_prediction_snapshot(team_df, bracket_df, "2026-06-10", "pre_tournament", 100)
    with pytest.raises(FileExistsError):
        save_prediction_snapshot(team_df, bracket_df, "2026-06-10", "pre_tournament", 100)
    save_prediction_snapshot(team_df, bracket_df, "2026-06-10", "pre_tournament", 100, force=True)


def test_load_actual_results_empty(tmp_path):
    p = tmp_path / "actuals.csv"
    pd.DataFrame(columns=["stage", "team_a", "team_b", "score_a", "score_b", "winner"]).to_csv(p, index=False)
    assert load_actual_results(str(p)) == {}


def test_load_actual_results_parses(tmp_path):
    p = tmp_path / "actuals.csv"
    pd.DataFrame([
        {"stage": "Group A", "team_a": "USA", "team_b": "Mexico",
         "score_a": 2, "score_b": 1, "winner": "USA"},
    ]).to_csv(p, index=False)
    kr = load_actual_results(str(p))
    key = frozenset({"United States", "Mexico"})
    assert key in kr
    assert kr[key]["winner"] == "United States"
    assert kr[key]["score_a"] == 2 and kr[key]["team_a"] == "United States"
