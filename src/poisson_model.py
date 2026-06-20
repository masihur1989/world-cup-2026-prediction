import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

POISSON_FEATURES_A = [
    "elo_diff", "form_A", "goals_scored_avg_A",
    "fifa_rank_diff", "h2h_goal_diff", "stage_weight",
]
POISSON_FEATURES_B = [
    "elo_diff_neg", "form_B", "goals_scored_avg_B",
    "fifa_rank_diff_neg", "h2h_goal_diff_neg", "stage_weight",
]


class PoissonGoalModel:
    """
    Two Poisson GLMs (log link) predicting expected goals per team.
    """

    def __init__(self):
        self.model_a = None
        self.model_b = None

    def _prep_X(self, features: pd.DataFrame, for_team: str) -> pd.DataFrame:
        df = features.copy()
        df["elo_diff_neg"]       = -df["elo_diff"]
        df["fifa_rank_diff_neg"] = -df["fifa_rank_diff"]
        df["h2h_goal_diff_neg"]  = -df["h2h_goal_diff"]
        cols = POISSON_FEATURES_A if for_team == "A" else POISSON_FEATURES_B
        X = df[cols].fillna(0.0)
        return sm.add_constant(X, has_constant="add")

    def fit(self, features: pd.DataFrame) -> None:
        if "home_score" in features.columns:
            y_a = features["home_score"].fillna(features["goals_scored_avg_A"].fillna(1.5))
            y_b = features["away_score"].fillna(features["goals_scored_avg_B"].fillna(1.2))
        else:
            y_a = features["goals_scored_avg_A"].fillna(1.5)
            y_b = features["goals_scored_avg_B"].fillna(1.2)

        X_a = self._prep_X(features, "A")
        X_b = self._prep_X(features, "B")

        self.model_a = sm.GLM(y_a, X_a, family=sm.families.Poisson()).fit()
        self.model_b = sm.GLM(y_b, X_b, family=sm.families.Poisson()).fit()

    def predict_lambda(self, row: pd.Series) -> tuple[float, float]:
        df = pd.DataFrame([row])
        X_a = self._prep_X(df, "A")
        X_b = self._prep_X(df, "B")
        lambda_a = float(self.model_a.predict(X_a).iloc[0])
        lambda_b = float(self.model_b.predict(X_b).iloc[0])
        return max(lambda_a, 0.1), max(lambda_b, 0.1)

    def simulate_match(
        self, lambda_a: float, lambda_b: float, n: int = 50_000
    ) -> tuple[float, float, float]:
        rng = np.random.default_rng(42)
        goals_a = rng.poisson(lambda_a, n)
        goals_b = rng.poisson(lambda_b, n)
        p_win  = float(np.mean(goals_a > goals_b))
        p_draw = float(np.mean(goals_a == goals_b))
        p_loss = float(np.mean(goals_a < goals_b))
        return p_win, p_draw, p_loss

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model_a": self.model_a, "model_b": self.model_b}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model_a = data["model_a"]
        self.model_b = data["model_b"]


def add_poisson_features(
    features: pd.DataFrame, model: PoissonGoalModel
) -> pd.DataFrame:
    """Append lambda_a and p_win_poisson columns (features 17 and 18)."""
    features = features.copy()
    lambdas, p_wins = [], []
    for _, row in features.iterrows():
        la, lb = model.predict_lambda(row)
        pw, _, _ = model.simulate_match(la, lb, n=50_000)
        lambdas.append(la)
        p_wins.append(pw)
    features["lambda_a"]      = lambdas
    features["p_win_poisson"] = p_wins
    return features
