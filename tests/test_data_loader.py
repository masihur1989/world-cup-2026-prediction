import pandas as pd
import pytest
from src.data_loader import load_results, load_shootouts, load_fixtures


def test_load_results_returns_dataframe(tmp_path, mini_results):
    path = tmp_path / "results.csv"
    mini_results.to_csv(path, index=False)
    result = load_results(path)
    assert isinstance(result, pd.DataFrame)
    required_cols = {"date", "home_team", "away_team", "home_score", "away_score", "tournament"}
    assert required_cols.issubset(result.columns)


def test_load_results_filters_to_1990(tmp_path, mini_results):
    old_row = mini_results.copy()
    old_row.iloc[0, old_row.columns.get_loc("date")] = pd.Timestamp("1985-01-01")
    extended = pd.concat([old_row, mini_results], ignore_index=True)
    path = tmp_path / "results.csv"
    extended.to_csv(path, index=False)
    result = load_results(path)
    assert (result["date"] >= "1990-01-01").all()


def test_load_results_sorted_by_date(tmp_path, mini_results):
    shuffled = mini_results.sample(frac=1, random_state=42).reset_index(drop=True)
    path = tmp_path / "results.csv"
    shuffled.to_csv(path, index=False)
    result = load_results(path)
    assert result["date"].is_monotonic_increasing


def test_load_shootouts_has_winner_column(tmp_path, mini_shootouts):
    path = tmp_path / "shootouts.csv"
    mini_shootouts.to_csv(path, index=False)
    result = load_shootouts(path)
    assert "winner" in result.columns


def test_load_fixtures_has_required_columns(tmp_path, mini_fixtures):
    path = tmp_path / "fixtures.csv"
    mini_fixtures.to_csv(path, index=False)
    result = load_fixtures(path)
    required = {"team_a", "team_b", "stage", "date", "venue"}
    assert required.issubset(result.columns)
