import pandas as pd
import numpy as np
from typing import Optional

CONTINENTAL_TOURNAMENTS = {
    "UEFA Euro", "Copa America", "Africa Cup of Nations",
    "AFC Asian Cup", "CONCACAF Gold Cup", "OFC Nations Cup",
    "UEFA Euro qualification", "Copa America qualification",
    "Africa Cup of Nations qualification", "AFC Asian Cup qualification",
    "CONCACAF Gold Cup qualification", "OFC Nations Cup qualification",
}

def K_FACTOR(tournament: str) -> int:
    if tournament == "FIFA World Cup":
        return 40
    if tournament in CONTINENTAL_TOURNAMENTS:
        return 30
    return 20

def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def compute_elo_history(results: pd.DataFrame) -> pd.DataFrame:
    """
    Chronologically iterate results, updating Elo after every match.
    Returns DataFrame with (date, team, elo_rating) where elo_rating
    is the team's rating AFTER the match on that date.
    """
    elo: dict[str, float] = {}
    records = []

    for _, row in results.iterrows():
        home, away = row["home_team"], row["away_team"]
        elo.setdefault(home, 1500.0)
        elo.setdefault(away, 1500.0)

        elo_h, elo_a = elo[home], elo[away]
        exp_h = _expected_score(elo_h, elo_a)
        exp_a = 1.0 - exp_h

        if row["home_score"] > row["away_score"]:
            actual_h, actual_a = 1.0, 0.0
        elif row["home_score"] == row["away_score"]:
            actual_h, actual_a = 0.5, 0.5
        else:
            actual_h, actual_a = 0.0, 1.0

        k = K_FACTOR(row["tournament"])
        elo[home] = elo_h + k * (actual_h - exp_h)
        elo[away] = elo_a + k * (actual_a - exp_a)

        records.append({"date": row["date"], "team": home, "elo_rating": elo[home]})
        records.append({"date": row["date"], "team": away, "elo_rating": elo[away]})

    history = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return history

def get_elo_on_date(
    elo_history: pd.DataFrame, team: str, date: pd.Timestamp
) -> float:
    """Return most recent Elo rating for team strictly before date."""
    subset = elo_history[
        (elo_history["team"] == team) & (elo_history["date"] < date)
    ]
    if subset.empty:
        return 1500.0
    return float(subset.sort_values("date").iloc[-1]["elo_rating"])

def get_rank_on_date(
    elo_history: pd.DataFrame, date: pd.Timestamp, team: str,
    active_window_days: int = 365
) -> int:
    """
    Rank team among all active teams (played >=1 match in prior 12 months)
    by Elo descending on date. Rank 1 = highest Elo = best.
    """
    cutoff = date - pd.Timedelta(days=active_window_days)
    active_teams = elo_history[
        (elo_history["date"] >= cutoff) & (elo_history["date"] < date)
    ]["team"].unique()

    if team not in active_teams:
        return 999

    latest = (
        elo_history[elo_history["date"] < date]
        .sort_values("date")
        .groupby("team")["elo_rating"]
        .last()
        .reindex(active_teams)
        .fillna(1500.0)
        .sort_values(ascending=False)
        .reset_index()
    )
    latest.index = latest.index + 1
    latest.index.name = "rank"
    rank_row = latest[latest["team"] == team]
    if rank_row.empty:
        return 999
    return int(rank_row.index[0])

def save_elo_history(elo_history: pd.DataFrame, path: str = "data/processed/elo_history.csv") -> None:
    elo_history.to_csv(path, index=False)
