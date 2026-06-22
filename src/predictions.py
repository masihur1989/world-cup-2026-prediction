from datetime import datetime
from pathlib import Path

import pandas as pd

from src.fixtures_bracket import normalize_team

PROCESSED = Path("data/processed")
PREDICTIONS_DIR = PROCESSED / "predictions"


def save_prediction_snapshot(team_df, bracket_df, as_of_date, label,
                             n_simulations, matchups_df=None, force=False):
    """
    Write a dated snapshot (team table + bracket frame, plus an optional per-pair
    matchup-probability table), append to the index, and refresh the 'latest'
    files the dashboard reads by default. Returns dict of written paths. Raises
    FileExistsError if the snapshot exists and force is False.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{as_of_date}__{label}"
    team_path = PREDICTIONS_DIR / f"{stem}.csv"
    bracket_path = PREDICTIONS_DIR / f"{stem}__bracket.csv"
    matchups_path = PREDICTIONS_DIR / f"{stem}__matchups.csv"

    if team_path.exists() and not force:
        raise FileExistsError(f"Snapshot already exists: {team_path} (use force=True)")

    team_df.to_csv(team_path, index=False)
    bracket_df.to_csv(bracket_path, index=False)
    if matchups_df is not None:
        matchups_df.to_csv(matchups_path, index=False)

    index_path = PREDICTIONS_DIR / "index.csv"
    row = {"as_of_date": as_of_date, "label": label, "n_simulations": n_simulations,
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "path": team_path.name, "bracket_path": bracket_path.name,
           "matchups_path": matchups_path.name if matchups_df is not None else ""}
    if index_path.exists():
        idx = pd.read_csv(index_path)
        idx = idx[~((idx["as_of_date"].astype(str) == str(as_of_date)) &
                    (idx["label"] == label))]
        idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
    else:
        idx = pd.DataFrame([row])
    idx.to_csv(index_path, index=False)

    team_df.to_csv(PROCESSED / "simulation_results.csv", index=False)
    bracket_df.to_csv(PROCESSED / "bracket.csv", index=False)
    if matchups_df is not None:
        matchups_df.to_csv(PROCESSED / "matchups.csv", index=False)

    return {"team": team_path, "bracket": bracket_path, "index": index_path}


def load_actual_results(path="data/raw/wc2026_actual_results.csv") -> dict:
    """
    Read the actual-results CSV into a known_results override dict keyed by
    frozenset({team_a, team_b}) with normalized names. Missing/empty file -> {}.
    """
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    out: dict = {}
    for _, r in df.iterrows():
        a = normalize_team(r["team_a"]); b = normalize_team(r["team_b"])
        winner = normalize_team(r["winner"]) if pd.notna(r.get("winner")) else None
        sa = r.get("score_a"); sb = r.get("score_b")
        out[frozenset({a, b})] = {
            "team_a": a, "team_b": b,
            "score_a": None if pd.isna(sa) else int(sa),
            "score_b": None if pd.isna(sb) else int(sb),
            "winner": winner,
        }
    return out
