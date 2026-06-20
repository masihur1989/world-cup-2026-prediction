import numpy as np
import pandas as pd
import pytest
from src.fixtures_bracket import load_tournament
from src.simulator import TournamentSimulator, get_8_best_third
from src.xgboost_model import FEATURE_COLS


@pytest.fixture
def tournament():
    return load_tournament("data/raw/wc2026_fixtures.csv")


class StubModel:
    def predict_proba(self, X):
        n = len(X)
        return np.full((n, 3), 1 / 3)
    label_encoder = type("LE", (), {"classes_": np.array([-1, 0, 1])})()


class StubProvider:
    def row(self, a, b):
        return pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}])


def make_sim(tournament, **kw):
    teams = [t for g in tournament.groups.values() for t in g]
    penalty = {t: 0.5 for t in teams}
    return TournamentSimulator(StubModel(), penalty, tournament, StubProvider(), **kw)


def test_get_8_best_third_unchanged():
    thirds = [{"team": f"T{i}", "pts": i % 5, "gd": i % 3, "gf": i % 7} for i in range(12)]
    assert len(get_8_best_third(thirds)) == 8


def test_simulate_group_returns_4_standings(tournament):
    sim = make_sim(tournament)
    standings = sim.simulate_group("A")
    assert len(standings) == 4
    assert {"team", "pts", "gd", "gf"} <= set(standings[0])


def test_full_tournament_has_champion(tournament):
    sim = make_sim(tournament)
    res = sim.simulate_full_tournament()
    teams = [t for g in tournament.groups.values() for t in g]
    assert res["champion"] in teams
    assert len(res["semifinalists"]) == 4
    assert len(res["finalists"]) == 2


def test_run_outputs_advancement_columns(tournament):
    sim = make_sim(tournament)
    df, _bracket = sim.run(n_simulations=50)
    assert len(df) == 48
    for col in ["p_champion", "p_finalist", "p_semifinalist",
                "p_group_winner", "p_runner_up", "p_advance", "confederation"]:
        assert col in df.columns


def test_champion_probs_sum_to_one(tournament):
    sim = make_sim(tournament)
    df, _ = sim.run(n_simulations=200)
    assert abs(df["p_champion"].sum() - 1.0) < 0.01


def test_advance_count_is_32_per_sim(tournament):
    sim = make_sim(tournament)
    df, _ = sim.run(n_simulations=100)
    # 12 winners + 12 runners + 8 best thirds = 32 teams advance each sim
    assert (df["p_advance"] * 1).sum() == pytest.approx(32.0, abs=0.5)


def test_bracket_frame_positions(tournament):
    sim = make_sim(tournament)
    _df, bracket = sim.run(n_simulations=100)
    assert set(bracket.columns) == {"round", "match_index", "team", "p_reach"}
    counts = bracket.groupby("round")["match_index"].nunique().to_dict()
    assert counts["R32"] == 16
    assert counts["R16"] == 8
    assert counts["QF"] == 4
    assert counts["SF"] == 2
    assert counts["Final"] == 1


def test_bracket_two_participants_per_match(tournament):
    sim = make_sim(tournament)
    _df, bracket = sim.run(n_simulations=200)
    s = bracket.groupby(["round", "match_index"])["p_reach"].sum()
    assert np.allclose(s.values, 2.0, atol=0.05)
