import pandas as pd
import pytest
from src.elo_calculator import (
    compute_elo_history,
    get_elo_on_date,
    get_rank_on_date,
    K_FACTOR,
    CONTINENTAL_TOURNAMENTS,
)

def test_k_factor_world_cup():
    assert K_FACTOR("FIFA World Cup") == 40

def test_k_factor_continental():
    assert K_FACTOR("UEFA Euro") == 30
    assert K_FACTOR("Copa America") == 30
    assert K_FACTOR("Africa Cup of Nations") == 30

def test_k_factor_other():
    assert K_FACTOR("Friendly") == 20
    assert K_FACTOR("FIFA World Cup qualification") == 20

def test_elo_history_shape(mini_results):
    history = compute_elo_history(mini_results)
    assert isinstance(history, pd.DataFrame)
    assert set(history.columns) >= {"date", "team", "elo_rating"}

def test_all_teams_present_in_history(mini_results):
    history = compute_elo_history(mini_results)
    teams_in_matches = set(mini_results["home_team"]) | set(mini_results["away_team"])
    teams_in_history = set(history["team"])
    assert teams_in_matches == teams_in_history

def test_elo_initialized_at_1500(mini_results):
    history = compute_elo_history(mini_results)
    germany_first = history[history["team"] == "Germany"].sort_values("date").iloc[0]
    assert germany_first["elo_rating"] != 1500

def test_winner_elo_increases(mini_results):
    history = compute_elo_history(mini_results)
    germany_snap = history[(history["team"] == "Germany") & (history["date"] == pd.Timestamp("1990-06-01"))]["elo_rating"].values[0]
    france_snap  = history[(history["team"] == "France")  & (history["date"] == pd.Timestamp("1990-06-01"))]["elo_rating"].values[0]
    assert germany_snap > 1500
    assert france_snap  < 1500

def test_elo_history_sorted_by_date(mini_results):
    history = compute_elo_history(mini_results)
    assert history["date"].is_monotonic_increasing

def test_get_elo_on_date(mini_results):
    history = compute_elo_history(mini_results)
    elo = get_elo_on_date(history, "Germany", pd.Timestamp("1990-06-02"))
    assert isinstance(elo, float)
    assert 1400 < elo < 1600

def test_get_elo_on_date_unknown_team(mini_results):
    history = compute_elo_history(mini_results)
    elo = get_elo_on_date(history, "Atlantis", pd.Timestamp("2020-01-01"))
    assert elo == 1500.0

def test_get_rank_on_date(mini_results):
    history = compute_elo_history(mini_results)
    rank = get_rank_on_date(history, pd.Timestamp("2022-11-24"), "Germany")
    assert isinstance(rank, int)
    assert 1 <= rank <= 4
