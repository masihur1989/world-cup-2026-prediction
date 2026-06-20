import numpy as np
import pandas as pd
import pytest
from src.fixtures_bracket import load_tournament
from src.simulator import TournamentSimulator
from src.xgboost_model import FEATURE_COLS


class StubModel:
    def predict_proba(self, X):
        return np.full((len(X), 3), 1 / 3)
    label_encoder = type("LE", (), {"classes_": np.array([-1, 0, 1])})()


class StubProvider:
    def row(self, a, b):
        return pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}])


@pytest.fixture
def tournament():
    return load_tournament("data/raw/wc2026_fixtures.csv")


def test_known_group_result_fixes_score(tournament):
    a, b, _g = tournament.group_matches[0]
    known = {frozenset({a, b}): {"team_a": a, "team_b": b,
                                 "score_a": 3, "score_b": 0, "winner": a}}
    teams = [t for grp in tournament.groups.values() for t in grp]
    sim = TournamentSimulator(StubModel(), {t: 0.5 for t in teams},
                              tournament, StubProvider(), known_results=known)
    for _ in range(10):
        assert sim.simulate_group_match(a, b) == (3, 0)


def test_known_knockout_result_fixes_winner(tournament):
    teams = [t for grp in tournament.groups.values() for t in grp]
    a, b = teams[0], teams[1]
    known = {frozenset({a, b}): {"team_a": a, "team_b": b,
                                 "score_a": None, "score_b": None, "winner": b}}
    sim = TournamentSimulator(StubModel(), {t: 0.5 for t in teams},
                              tournament, StubProvider(), known_results=known)
    for _ in range(10):
        assert sim.simulate_knockout_match(a, b) == b
