import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def mini_results():
    """10 matches, 4 teams, chronological, enough for rolling features."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "1990-06-01", "1990-06-02", "1990-06-03", "1990-06-10",
            "1990-06-15", "1991-01-01", "1991-06-01", "1991-06-10",
            "2022-11-20", "2022-11-24",
        ]),
        "home_team": ["Germany", "France", "Germany", "Brazil",
                      "France",  "Germany", "Brazil",  "France",
                      "Germany", "Brazil"],
        "away_team": ["France",  "Brazil",  "Brazil",  "France",
                      "Germany", "Brazil",  "France",  "Germany",
                      "Brazil",  "France"],
        "home_score": [1, 2, 3, 1, 0, 2, 1, 1, 2, 0],
        "away_score": [0, 1, 0, 1, 1, 0, 0, 2, 1, 2],
        "tournament": [
            "FIFA World Cup", "FIFA World Cup", "FIFA World Cup",
            "UEFA Euro", "Friendly",
            "FIFA World Cup", "Copa America", "Friendly",
            "FIFA World Cup", "FIFA World Cup",
        ],
        "city": ["Berlin"] * 10,
        "country": ["Germany"] * 10,
        "neutral": [False] * 10,
    })

@pytest.fixture
def mini_shootouts():
    return pd.DataFrame({
        "date": pd.to_datetime(["1990-06-01", "1991-01-01"]),
        "home_team": ["Germany", "Brazil"],
        "away_team": ["France",  "France"],
        "winner":    ["Germany", "France"],
    })

@pytest.fixture
def mini_fixtures():
    return pd.DataFrame({
        "team_a": ["Germany", "Brazil"],
        "team_b": ["France",  "France"],
        "stage":  ["Group A", "Group B"],
        "date":   pd.to_datetime(["2026-06-20", "2026-06-21"]),
        "venue":  ["MetLife Stadium", "AT&T Stadium"],
    })
