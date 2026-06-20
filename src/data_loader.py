from pathlib import Path
import pandas as pd

DATA_RAW = Path("data/raw")


def load_results(path: Path | str = DATA_RAW / "results.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] >= "1990-01-01"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_goalscorers(path: Path | str = DATA_RAW / "goalscorers.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_shootouts(path: Path | str = DATA_RAW / "shootouts.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_fixtures(path: Path | str = DATA_RAW / "wc2026_fixtures.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])
