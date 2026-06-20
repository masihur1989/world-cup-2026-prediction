import numpy as np
import pandas as pd
from collections import defaultdict

from src.feature_builder import CONFEDERATION
from src.fixtures_bracket import assign_third_place


def get_8_best_third(third_place: list[dict]) -> list[str]:
    """Select 8 best 3rd-place teams by pts desc, gd desc, gf desc."""
    ordered = sorted(third_place, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    return [t["team"] for t in ordered[:8]]


class TournamentSimulator:
    """
    Fixture-driven Monte Carlo simulator. Groups and R32 pairings come from a
    parsed Tournament; matchup features come from a provider (so knockout odds
    use real current-state features). Probabilities for every ordered pair are
    precomputed once, after which the simulation loop makes no model calls.
    """

    def __init__(self, model, penalty_rates, tournament, matchup_features,
                 known_results=None):
        self.model = model
        self.penalty_rates = penalty_rates
        self.tournament = tournament
        self.matchup_features = matchup_features
        self.known_results = known_results or {}
        self.teams = [t for g in tournament.groups.values() for t in g]
        self._prob_cache: dict = {}

    def precompute_probabilities(self) -> None:
        pairs = [(a, b) for a in self.teams for b in self.teams if a != b]
        X = pd.concat([self.matchup_features.row(a, b) for a, b in pairs],
                      ignore_index=True)
        proba = self.model.predict_proba(X)
        self._prob_cache = {pair: proba[i] for i, pair in enumerate(pairs)}

    def _proba(self, a, b):
        cached = self._prob_cache.get((a, b))
        if cached is not None:
            return cached
        return self.model.predict_proba(self.matchup_features.row(a, b))[0]

    def _class_probs(self, a, b):
        proba = self._proba(a, b)
        classes = list(self.model.label_encoder.classes_)
        m = {int(c): p for c, p in zip(classes, proba)}
        return m.get(1, 0.0), m.get(0, 0.0), m.get(-1, 0.0), len(classes)

    def simulate_group_match(self, a, b) -> tuple:
        known = self.known_results.get(frozenset({a, b}))
        if known is not None and known.get("score_a") is not None:
            if known["team_a"] == a:
                return int(known["score_a"]), int(known["score_b"])
            return int(known["score_b"]), int(known["score_a"])
        p_win_a, p_draw, _p_loss, _ = self._class_probs(a, b)
        r = np.random.random()
        if r < p_win_a:
            return (2, 0)
        if r < p_win_a + p_draw:
            return (1, 1)
        return (0, 2)

    def simulate_group(self, group: str) -> list:
        teams = self.tournament.groups[group]
        pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
        matches = [(a, b) for (a, b, g) in self.tournament.group_matches if g == group]
        for a, b in matches:
            ga, gb = self.simulate_group_match(a, b)
            gf[a] += ga; gf[b] += gb
            gd[a] += ga - gb; gd[b] += gb - ga
            if ga > gb:
                pts[a] += 3
            elif ga == gb:
                pts[a] += 1; pts[b] += 1
            else:
                pts[b] += 3
        standings = [{"team": t, "pts": pts[t], "gd": gd[t], "gf": gf[t]} for t in teams]
        standings.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
        return standings

    def simulate_knockout_match(self, a, b) -> str:
        known = self.known_results.get(frozenset({a, b}))
        if known is not None and known.get("winner"):
            return known["winner"]
        p_win_a, p_draw, _p_loss, n_classes = self._class_probs(a, b)
        if n_classes == 3:
            p_a = p_win_a + p_draw * 0.5
        else:
            p_a = p_win_a
        if abs(p_a - 0.5) < 0.01:
            pa = self.penalty_rates.get(a, 0.5)
            pb = self.penalty_rates.get(b, 0.5)
            p_a = pa / (pa + pb) if (pa + pb) > 0 else 0.5
        return a if np.random.random() < p_a else b

    def simulate_full_tournament(self) -> dict:
        group_winner, group_runner = {}, {}
        thirds_by_group = {}
        for g in self.tournament.groups:
            st = self.simulate_group(g)
            group_winner[g] = st[0]["team"]
            group_runner[g] = st[1]["team"]
            thirds_by_group[g] = {"group": g, **st[2]}

        best = get_8_best_third(list(thirds_by_group.values()))
        best_set = set(best)
        best_thirds = [(g, thirds_by_group[g]["team"]) for g in self.tournament.groups
                       if thirds_by_group[g]["team"] in best_set]

        third_slot_eligibles = []
        slot_lookup = []
        for i, (sa, sb) in enumerate(self.tournament.r32_slots):
            for side, s in (("a", sa), ("b", sb)):
                if s.kind == "third":
                    third_slot_eligibles.append(s.eligible)
                    slot_lookup.append((i, side))
        assigned = assign_third_place(third_slot_eligibles, best_thirds)
        third_by_pos = {pos: team for pos, team in zip(slot_lookup, assigned)}

        r32_winners = []
        advanced = set()
        for i, (sa, sb) in enumerate(self.tournament.r32_slots):
            ta = (third_by_pos[(i, "a")] if sa.kind == "third"
                  else (group_winner[sa.group] if sa.kind == "winner" else group_runner[sa.group]))
            tb = (third_by_pos[(i, "b")] if sb.kind == "third"
                  else (group_winner[sb.group] if sb.kind == "winner" else group_runner[sb.group]))
            advanced.add(ta); advanced.add(tb)
            r32_winners.append(self.simulate_knockout_match(ta, tb))

        current = r32_winners
        semifinalists, finalists = [], []
        while len(current) > 1:
            if len(current) == 4:
                semifinalists = list(current)
            if len(current) == 2:
                finalists = list(current)
            nxt = [self.simulate_knockout_match(current[i], current[i + 1])
                   for i in range(0, len(current), 2)]
            current = nxt

        return {
            "champion": current[0],
            "finalists": finalists,
            "semifinalists": semifinalists,
            "group_winners": list(group_winner.values()),
            "runners_up": list(group_runner.values()),
            "advanced": advanced,
        }

    def run(self, n_simulations: int = 100_000, progress_callback=None):
        if not self._prob_cache:
            self.precompute_probabilities()
        champ = defaultdict(int); fin = defaultdict(int); semi = defaultdict(int)
        gw = defaultdict(int); ru = defaultdict(int); adv = defaultdict(int)
        report_every = max(1, n_simulations // 20)
        for i in range(n_simulations):
            r = self.simulate_full_tournament()
            champ[r["champion"]] += 1
            for t in r["finalists"]:
                fin[t] += 1
            for t in r["semifinalists"]:
                semi[t] += 1
            for t in r["group_winners"]:
                gw[t] += 1
            for t in r["runners_up"]:
                ru[t] += 1
            for t in r["advanced"]:
                adv[t] += 1
            if progress_callback and (i + 1) % report_every == 0:
                progress_callback(i + 1, n_simulations)

        rows = [{
            "team": t,
            "p_champion": champ[t] / n_simulations,
            "p_finalist": fin[t] / n_simulations,
            "p_semifinalist": semi[t] / n_simulations,
            "p_group_winner": gw[t] / n_simulations,
            "p_runner_up": ru[t] / n_simulations,
            "p_advance": adv[t] / n_simulations,
            "confederation": CONFEDERATION.get(t, "UEFA"),
        } for t in self.teams]
        df = pd.DataFrame(rows)
        total = df["p_champion"].sum()
        if total > 0:
            df["p_champion"] = df["p_champion"] / total
        df = df.sort_values("p_champion", ascending=False).reset_index(drop=True)
        bracket = pd.DataFrame(columns=["round", "match_index", "team", "p_reach"])
        return df, bracket


def save_simulation_results(df: pd.DataFrame,
                            path: str = "data/processed/simulation_results.csv") -> None:
    df.to_csv(path, index=False)
