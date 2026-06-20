import numpy as np
import pandas as pd
import pytest
from src.feature_builder import MatchupFeatureProvider
from src.elo_calculator import compute_elo_history
from src.xgboost_model import FEATURE_COLS


@pytest.fixture
def mini():
    rng = np.random.default_rng(0)
    teams = ["Brazil", "France", "Spain", "Germany"]
    n = 120
    res = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=n, freq="10D"),
        "home_team": rng.choice(teams, n),
        "away_team": rng.choice(teams, n),
        "home_score": rng.integers(0, 4, n),
        "away_score": rng.integers(0, 4, n),
        "tournament": ["Friendly"] * n,
    })
    res = res[res["home_team"] != res["away_team"]].reset_index(drop=True)
    shoot = pd.DataFrame({"date": [], "home_team": [], "away_team": [], "winner": []})
    elo = compute_elo_history(res)
    return res, shoot, elo


def test_row_has_all_feature_cols(mini):
    res, shoot, elo = mini
    prov = MatchupFeatureProvider(elo, res, shoot, as_of_date=pd.Timestamp("2026-06-10"))
    row = prov.row("Brazil", "France")
    assert list(row.columns) == FEATURE_COLS
    assert len(row) == 1


def test_row_is_finite(mini):
    res, shoot, elo = mini
    prov = MatchupFeatureProvider(elo, res, shoot, as_of_date=pd.Timestamp("2026-06-10"))
    row = prov.row("Brazil", "France")
    assert np.isfinite(row.to_numpy(dtype=float)).all()


def test_elo_diff_antisymmetric(mini):
    res, shoot, elo = mini
    prov = MatchupFeatureProvider(elo, res, shoot, as_of_date=pd.Timestamp("2026-06-10"))
    ab = prov.row("Brazil", "France")["elo_diff"].iloc[0]
    ba = prov.row("France", "Brazil")["elo_diff"].iloc[0]
    assert ab == pytest.approx(-ba)


def test_unknown_team_falls_back(mini):
    res, shoot, elo = mini
    prov = MatchupFeatureProvider(elo, res, shoot, as_of_date=pd.Timestamp("2026-06-10"))
    row = prov.row("Brazil", "Atlantis")
    assert list(row.columns) == FEATURE_COLS
    assert np.isfinite(row.to_numpy(dtype=float)).all()
