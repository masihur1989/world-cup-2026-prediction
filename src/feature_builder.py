import pandas as pd
import numpy as np
from src.elo_calculator import get_elo_on_date, get_rank_on_date

# Hardcoded team → confederation for all 48 WC 2026 qualified nations.
# TODO: Verify against official FIFA 2026 qualification results.
CONFEDERATION: dict[str, str] = {
    # UEFA
    "Czechia": "UEFA", "Switzerland": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Scotland": "UEFA", "Turkey": "UEFA", "Germany": "UEFA", "Netherlands": "UEFA",
    "Sweden": "UEFA", "Belgium": "UEFA", "Spain": "UEFA", "France": "UEFA",
    "Norway": "UEFA", "Austria": "UEFA", "Portugal": "UEFA", "Croatia": "UEFA",
    "England": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    # CAF
    "South Africa": "CAF", "Morocco": "CAF", "Ivory Coast": "CAF", "Tunisia": "CAF",
    "Egypt": "CAF", "Cape Verde": "CAF", "Senegal": "CAF", "Algeria": "CAF",
    "DR Congo": "CAF", "Ghana": "CAF",
    # AFC
    "South Korea": "AFC", "Qatar": "AFC", "Australia": "AFC", "Japan": "AFC",
    "Iran": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC",
    # CONCACAF
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "Haiti": "CONCACAF",
    "Curacao": "CONCACAF", "United States": "CONCACAF", "Panama": "CONCACAF",
    # OFC
    "New Zealand": "OFC",
}

STAGE_WEIGHT: dict[str, int] = {
    "Group":        1,
    "Round of 32":  2,
    "Quarterfinal": 3,
    "Semifinal":    4,
    "Final":        5,
}

CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"]

_DECAY_WEIGHTS_6  = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_DECAY_WEIGHTS_10 = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])


def build_team_perspective(results: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level results into team-perspective rows (each match appears twice).
    Adds shift(1)-based rolling features per team: form, goals_scored_avg,
    goals_conceded_avg, last_match_date.
    """
    home = results[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "opponent", "gf", "ga"]
    home["is_home"] = True

    away = results[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "opponent", "gf", "ga"]
    away["is_home"] = False

    tp = pd.concat([home, away], ignore_index=True)
    tp["win"]    = (tp["gf"] > tp["ga"]).astype(int)
    tp["draw"]   = (tp["gf"] == tp["ga"]).astype(int)
    tp["loss"]   = (tp["gf"] < tp["ga"]).astype(int)
    tp["points"] = tp["win"] * 3 + tp["draw"]
    tp = tp.sort_values(["team", "date"]).reset_index(drop=True)

    def _rolling_stats(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("date").reset_index(drop=True)
        pts = grp["points"].shift(1)
        gf  = grp["gf"].shift(1)
        ga  = grp["ga"].shift(1)

        def _weighted_mean(s: pd.Series, w: np.ndarray) -> pd.Series:
            out = []
            for i in range(len(s)):
                vals = s.iloc[max(0, i - len(w)):i].dropna().values
                if len(vals) == 0:
                    out.append(np.nan)
                else:
                    n = len(vals)
                    weights = w[-n:]
                    out.append(np.average(vals, weights=weights))
            return pd.Series(out, index=s.index)

        grp["form"]               = pts.rolling(10, min_periods=1).sum()
        grp["goals_scored_avg"]   = _weighted_mean(gf, _DECAY_WEIGHTS_6)
        grp["goals_conceded_avg"] = _weighted_mean(ga, _DECAY_WEIGHTS_6)
        grp["last_match_date"]    = grp["date"].shift(1)
        return grp

    tp = tp.groupby("team", group_keys=False).apply(_rolling_stats)
    return tp.reset_index(drop=True)


def compute_penalty_win_rates(shootouts: pd.DataFrame) -> dict[str, float]:
    """Return dict of team → win rate in penalty shootouts."""
    rates: dict[str, dict] = {}
    for _, row in shootouts.iterrows():
        for team in [row["home_team"], row["away_team"]]:
            rates.setdefault(team, {"wins": 0, "total": 0})
            rates[team]["total"] += 1
        rates[row["winner"]]["wins"] += 1
    return {t: v["wins"] / v["total"] for t, v in rates.items()}


def _h2h_stats(
    results: pd.DataFrame, team_a: str, team_b: str, before_date: pd.Timestamp
) -> tuple[float, float]:
    """Return (win_rate_A, avg_goal_diff) from all prior A vs B meetings."""
    mask = (
        (
            ((results["home_team"] == team_a) & (results["away_team"] == team_b)) |
            ((results["home_team"] == team_b) & (results["away_team"] == team_a))
        ) &
        (results["date"] < before_date)
    )
    h2h = results[mask]
    if h2h.empty:
        return 0.5, 0.0

    wins_a, goal_diffs = 0, []
    for _, r in h2h.iterrows():
        if r["home_team"] == team_a:
            gd = r["home_score"] - r["away_score"]
        else:
            gd = r["away_score"] - r["home_score"]
        goal_diffs.append(gd)
        if gd > 0:
            wins_a += 1
        elif gd == 0:
            wins_a += 0.5
    return wins_a / len(h2h), float(np.mean(goal_diffs))


def _one_hot_conf(team: str, prefix: str) -> dict[str, int]:
    conf = CONFEDERATION.get(team, "UEFA")
    return {f"{prefix}_{c}": int(conf == c) for c in CONFEDERATIONS}


def _stage_from_tournament_or_stage(tournament: str, stage) -> int:
    """Map tournament name or explicit stage string to stage weight."""
    if stage is not None and not (isinstance(stage, float) and pd.isna(stage)):
        for key in STAGE_WEIGHT:
            if key.lower() in str(stage).lower():
                return STAGE_WEIGHT[key]
    return STAGE_WEIGHT["Group"]


def build_features(
    results: pd.DataFrame,
    elo_history: pd.DataFrame,
    shootouts: pd.DataFrame,
    fixtures: pd.DataFrame | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Build 16 base features for every match in results (and optionally fixtures).
    outcome column: 1=home_win, 0=draw, -1=away_win (NaN for fixtures).
    Features 17 (lambda_a) and 18 (p_win_poisson) added by poisson_model.py.
    """
    tp = build_team_perspective(results)
    penalty_rates = compute_penalty_win_rates(shootouts)

    # Pre-group team_perspective rows by team for faster prior-lookup
    tp_by_team = {team: grp.sort_values("date") for team, grp in tp.groupby("team")}

    def _lookup_stat(team: str, match_date: pd.Timestamp, col: str):
        grp = tp_by_team.get(team)
        if grp is None:
            return np.nan
        prior = grp[grp["date"] < match_date]
        if prior.empty:
            return np.nan
        return prior.iloc[-1][col]

    rows = []
    source_df = results
    if fixtures is not None:
        source_df = results.copy()
        fix_copy = fixtures.copy()
        if "home_team" not in fix_copy.columns:
            fix_copy = fix_copy.rename(columns={"team_a": "home_team", "team_b": "away_team"})
        fix_copy["home_score"] = np.nan
        fix_copy["away_score"] = np.nan
        if "tournament" not in fix_copy.columns:
            fix_copy["tournament"] = "FIFA World Cup"
        source_df = pd.concat([source_df, fix_copy], ignore_index=True)

    n_total = len(source_df)
    report_every = max(1, n_total // 20)  # ~5% increments
    for counter, (_, match) in enumerate(source_df.iterrows(), start=1):
        date    = match["date"]
        team_a  = match["home_team"]
        team_b  = match["away_team"]
        tournament = match.get("tournament", "FIFA World Cup")
        stage_col  = match.get("stage", None)

        elo_a  = get_elo_on_date(elo_history, team_a, date)
        elo_b  = get_elo_on_date(elo_history, team_b, date)
        rank_a = get_rank_on_date(elo_history, date, team_a)
        rank_b = get_rank_on_date(elo_history, date, team_b)

        form_a = _lookup_stat(team_a, date, "form")
        form_b = _lookup_stat(team_b, date, "form")
        gsa    = _lookup_stat(team_a, date, "goals_scored_avg")
        gsb    = _lookup_stat(team_b, date, "goals_scored_avg")
        gca    = _lookup_stat(team_a, date, "goals_conceded_avg")
        gcb    = _lookup_stat(team_b, date, "goals_conceded_avg")
        last_a = _lookup_stat(team_a, date, "last_match_date")
        last_b = _lookup_stat(team_b, date, "last_match_date")

        rest_a = (date - last_a).days if pd.notna(last_a) else 30
        rest_b = (date - last_b).days if pd.notna(last_b) else 30

        h2h_wr, h2h_gd = _h2h_stats(results, team_a, team_b, date)

        hs = match.get("home_score", np.nan)
        as_ = match.get("away_score", np.nan)
        if pd.notna(hs) and pd.notna(as_):
            if hs > as_:
                outcome = 1
            elif hs == as_:
                outcome = 0
            else:
                outcome = -1
        else:
            outcome = np.nan

        row = {
            "date":      date,
            "team_a":    team_a,
            "team_b":    team_b,
            "tournament": tournament,
            "home_score": hs,
            "away_score": as_,
            "elo_diff":              elo_a - elo_b,
            "fifa_rank_diff":        rank_b - rank_a,
            "form_A":                form_a if pd.notna(form_a) else 0.0,
            "form_B":                form_b if pd.notna(form_b) else 0.0,
            "goals_scored_avg_A":    gsa    if pd.notna(gsa)    else 1.5,
            "goals_scored_avg_B":    gsb    if pd.notna(gsb)    else 1.5,
            "goals_conceded_avg_A":  gca    if pd.notna(gca)    else 1.2,
            "goals_conceded_avg_B":  gcb    if pd.notna(gcb)    else 1.2,
            "h2h_win_rate_A":        h2h_wr,
            "h2h_goal_diff":         h2h_gd,
            # Feature 11 — squad_value_ratio always 0.0
            # TODO: integrate Transfermarkt squad value data when available
            "squad_value_ratio":     0.0,
            "stage_weight":          _stage_from_tournament_or_stage(tournament, stage_col),
            "rest_days_diff":        rest_a - rest_b,
            "penalty_win_rate_A":    penalty_rates.get(team_a, 0.5),
            "outcome":               outcome,
            **_one_hot_conf(team_a, "conf_A"),
            **_one_hot_conf(team_b, "conf_B"),
        }
        rows.append(row)
        if progress_callback and counter % report_every == 0:
            progress_callback(counter, n_total)

    if progress_callback and n_total:
        progress_callback(n_total, n_total)
    return pd.DataFrame(rows)


def save_features(features: pd.DataFrame, path: str = "data/processed/features.csv") -> None:
    features.to_csv(path, index=False)
