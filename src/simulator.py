import numpy as np
import pandas as pd
from collections import defaultdict
from src.feature_builder import CONFEDERATION
from src.xgboost_model import FEATURE_COLS

WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["United States", "Brazil",    "Morocco",    "Serbia"],
    "B": ["Mexico",        "Netherlands","Japan",      "Cameroon"],
    "C": ["Canada",        "Germany",   "Ecuador",    "Ivory Coast"],
    "D": ["Spain",         "Argentina", "Saudi Arabia","Albania"],
    "E": ["France",        "Colombia",  "South Korea","Tunisia"],
    "F": ["England",       "Uruguay",   "Iran",       "DR Congo"],
    "G": ["Portugal",      "Belgium",   "Australia",  "Senegal"],
    "H": ["Croatia",       "Venezuela", "Egypt",      "Iraq"],
    "I": ["Austria",       "Paraguay",  "Jordan",     "Mali"],
    "J": ["Denmark",       "Turkey",    "Nigeria",    "Panama"],
    "K": ["Switzerland",   "Scotland",  "Uzbekistan", "Costa Rica"],
    "L": ["Czechia",       "Slovakia",  "New Zealand","Honduras"],
}


def get_8_best_third(third_place: list[dict]) -> list[str]:
    """Select 8 best 3rd-place teams by pts desc, gd desc, gf desc."""
    sorted_third = sorted(
        third_place,
        key=lambda x: (x["pts"], x["gd"], x["gf"]),
        reverse=True,
    )
    return [t["team"] for t in sorted_third[:8]]


class TournamentSimulator:
    def __init__(self, xgb_model, penalty_rates: dict, features: pd.DataFrame):
        self.model = xgb_model
        self.penalty_rates = penalty_rates
        self.features = features
        self._feature_cache: dict = {}

    def _get_features(self, team_a: str, team_b: str) -> pd.DataFrame:
        """Look up or generate feature row for team_a vs team_b."""
        cache_key = (team_a, team_b)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        mask = (self.features["team_a"] == team_a) & (self.features["team_b"] == team_b)
        if mask.any():
            row = self.features[mask].iloc[[-1]][FEATURE_COLS].fillna(0.0)
        else:
            mask_rev = (self.features["team_a"] == team_b) & (self.features["team_b"] == team_a)
            if mask_rev.any():
                row = self.features[mask_rev].iloc[[-1]][FEATURE_COLS].fillna(0.0).copy()
                for col in ["elo_diff", "fifa_rank_diff", "h2h_goal_diff"]:
                    if col in row.columns:
                        row[col] = -row[col]
            else:
                row = pd.DataFrame([{col: 0.0 for col in FEATURE_COLS}])
        self._feature_cache[cache_key] = row
        return row

    def simulate_group_match(self, team_a: str, team_b: str) -> tuple:
        X = self._get_features(team_a, team_b)
        proba = self.model.predict_proba(X)[0]
        classes = self.model.label_encoder.classes_
        class_probs = {int(c): p for c, p in zip(classes, proba)}
        p_win_a  = class_probs.get(1, 0.0)
        p_draw   = class_probs.get(0, 0.0)

        r = np.random.random()
        if r < p_win_a:
            return (2, 0)
        elif r < p_win_a + p_draw:
            return (1, 1)
        else:
            return (0, 2)

    def simulate_knockout_match(self, team_a: str, team_b: str) -> str:
        X = self._get_features(team_a, team_b)
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.label_encoder.classes_)

        if len(classes) == 3:
            p_a = proba[classes.index(1)] + proba[classes.index(0)] * 0.5
        else:
            idx_win = list(classes).index(1) if 1 in classes else -1
            p_a = proba[idx_win] if idx_win >= 0 else 0.5

        r = np.random.random()
        if abs(p_a - 0.5) < 0.01:
            pen_a = self.penalty_rates.get(team_a, 0.5)
            pen_b = self.penalty_rates.get(team_b, 0.5)
            total = pen_a + pen_b
            p_a = pen_a / total if total > 0 else 0.5

        return team_a if r < p_a else team_b

    def simulate_group(self, group: str) -> list:
        from itertools import combinations
        teams = WC2026_GROUPS[group]
        pts = defaultdict(int)
        gd  = defaultdict(int)
        gf  = defaultdict(int)

        for t_a, t_b in combinations(teams, 2):
            ga, gb = self.simulate_group_match(t_a, t_b)
            gf[t_a] += ga; gf[t_b] += gb
            gd[t_a] += ga - gb; gd[t_b] += gb - ga
            if ga > gb:
                pts[t_a] += 3
            elif ga == gb:
                pts[t_a] += 1; pts[t_b] += 1
            else:
                pts[t_b] += 3

        standings = [
            {"team": t, "pts": pts[t], "gd": gd[t], "gf": gf[t]}
            for t in teams
        ]
        standings.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
        return standings

    def simulate_full_tournament(self) -> dict:
        all_third: list = []
        qualified: dict = {}

        for group in WC2026_GROUPS:
            standings = self.simulate_group(group)
            qualified[group] = [standings[0]["team"], standings[1]["team"]]
            all_third.append(standings[2])

        best_third = get_8_best_third(all_third)

        group_keys = sorted(WC2026_GROUPS.keys())
        winners = [qualified[g][0] for g in group_keys]
        runners = [qualified[g][1] for g in group_keys]
        all_32 = winners + runners + best_third
        np.random.shuffle(all_32)
        ko_pairs = [(all_32[i], all_32[i+1]) for i in range(0, 32, 2)]

        current_round = [self.simulate_knockout_match(a, b) for a, b in ko_pairs]
        semifinalists = []
        finalists = []
        while len(current_round) > 1:
            if len(current_round) == 4:
                semifinalists = list(current_round)
            if len(current_round) == 2:
                finalists = list(current_round)
            next_round = []
            for i in range(0, len(current_round), 2):
                winner = self.simulate_knockout_match(
                    current_round[i], current_round[i+1]
                )
                next_round.append(winner)
            current_round = next_round

        return {
            "champion": current_round[0],
            "finalists": finalists,
            "semifinalists": semifinalists,
        }

    def run(self, n_simulations: int = 100_000) -> pd.DataFrame:
        all_teams = [t for group in WC2026_GROUPS.values() for t in group]
        champion_counts     = defaultdict(int)
        finalist_counts     = defaultdict(int)
        semifinalist_counts = defaultdict(int)

        for _ in range(n_simulations):
            res = self.simulate_full_tournament()
            champion_counts[res["champion"]] += 1
            for t in res["finalists"]:
                finalist_counts[t] += 1
            for t in res["semifinalists"]:
                semifinalist_counts[t] += 1

        results = []
        for team in all_teams:
            results.append({
                "team":           team,
                "p_champion":     champion_counts[team] / n_simulations,
                "p_finalist":     finalist_counts[team] / n_simulations,
                "p_semifinalist": semifinalist_counts[team] / n_simulations,
                "confederation":  CONFEDERATION.get(team, "UEFA"),
            })

        df = pd.DataFrame(results)
        total = df["p_champion"].sum()
        if total > 0:
            df["p_champion"] = df["p_champion"] / total
        return df.sort_values("p_champion", ascending=False).reset_index(drop=True)


def save_simulation_results(
    df: pd.DataFrame,
    path: str = "data/processed/simulation_results.csv",
) -> None:
    df.to_csv(path, index=False)
