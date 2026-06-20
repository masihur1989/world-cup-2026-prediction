import pickle
import numpy as np
import pandas as pd
import mlflow
import optuna
import shap
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("pipeline.xgboost")

SCALAR_FEATURES = [
    "elo_diff", "fifa_rank_diff", "form_A", "form_B",
    "goals_scored_avg_A", "goals_scored_avg_B",
    "goals_conceded_avg_A", "goals_conceded_avg_B",
    "h2h_win_rate_A", "h2h_goal_diff", "squad_value_ratio",
    "stage_weight", "rest_days_diff",
]
CONF_FEATURES = [
    f"conf_{side}_{c}"
    for side in ["A", "B"]
    for c in ["UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"]
]
FEATURE_COLS = SCALAR_FEATURES + CONF_FEATURES + ["penalty_win_rate_A", "lambda_a", "p_win_poisson"]


def compute_rps(y_true_onehot: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Ranked Probability Score — lower is better."""
    cum_true = np.cumsum(y_true_onehot, axis=1)
    cum_pred = np.cumsum(y_pred_proba, axis=1)
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1) / (y_true_onehot.shape[1] - 1)))


def compute_brier(y_true_onehot: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Mean Brier score across all classes."""
    return float(np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1)))


class WorldCupXGBModel:
    """
    mode='3class': predicts W/D/L (group stage)
    mode='binary': predicts W/L only, draw class suppressed (knockout stage)
    """

    def __init__(self, mode: str = "3class"):
        assert mode in {"3class", "binary"}
        self.mode = mode
        self.calibrated_model = None
        self.label_encoder = LabelEncoder()
        self._base_model = None

    def tune(
        self, X: pd.DataFrame, y: np.ndarray, dates: pd.Index, n_trials: int = 100
    ) -> dict:
        """Optuna hyperparameter search using TimeSeriesSplit CV."""
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
            }
            scores = []
            dates_arr = np.asarray(dates)
            sort_idx = np.argsort(dates_arr)
            X_s = X.iloc[sort_idx]
            y_s = np.asarray(y)[sort_idx]
            for train_idx, val_idx in tscv.split(X_s):
                X_tr, X_val = X_s.iloc[train_idx], X_s.iloc[val_idx]
                y_tr, y_val = y_s[train_idx], y_s[val_idx]
                xgb = XGBClassifier(tree_method="hist", **params)
                le = LabelEncoder()
                y_enc_tr = le.fit_transform(y_tr)
                xgb.fit(X_tr, y_enc_tr, verbose=False)
                proba = xgb.predict_proba(X_val)
                from sklearn.metrics import log_loss
                y_enc_val = le.transform(y_val)
                scores.append(log_loss(y_enc_val, proba, labels=np.arange(len(le.classes_))))
            return np.mean(scores)

        report_every = max(1, n_trials // 10)  # ~10% increments

        def _log_progress(study, trial):
            done = trial.number + 1
            if done % report_every == 0 or done == n_trials:
                logger.info(
                    "  tuning %s: trial %d/%d (%.0f%%) best_logloss=%.4f",
                    self.mode, done, n_trials, 100 * done / n_trials, study.best_value,
                )

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[_log_progress])
        return study.best_params

    def fit(
        self, X: pd.DataFrame, y: np.ndarray,
        dates: pd.Index, params: dict
    ) -> None:
        """Fit XGBoost with Platt scaling calibration via TimeSeriesSplit."""
        clean = {k: v for k, v in params.items() if k not in ("use_label_encoder", "eval_metric")}
        base = XGBClassifier(tree_method="hist", **clean)
        y_enc = self.label_encoder.fit_transform(y)
        tscv = TimeSeriesSplit(n_splits=5)
        self._base_model = base
        self.calibrated_model = CalibratedClassifierCV(base, cv=tscv, method="sigmoid")
        self.calibrated_model.fit(X, y_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability matrix — rows sum to 1.0 (guaranteed by calibration)."""
        return self.calibrated_model.predict_proba(X)

    def compute_shap(self, X: pd.DataFrame):
        """Compute SHAP values using the underlying XGBoost base estimator."""
        cc = self.calibrated_model.calibrated_classifiers_[0]
        base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        explainer = shap.TreeExplainer(base)
        return explainer.shap_values(X)

    def evaluate(
        self, X: pd.DataFrame, y_true: np.ndarray
    ) -> dict:
        """Return accuracy, log-loss, Brier, RPS on provided data."""
        from sklearn.metrics import accuracy_score, log_loss
        proba = self.predict_proba(X)
        y_enc = self.label_encoder.transform(y_true)
        classes = self.label_encoder.classes_
        n_classes = len(classes)

        y_onehot = np.zeros((len(y_enc), n_classes))
        for i, c in enumerate(y_enc):
            y_onehot[i, c] = 1.0

        return {
            "accuracy": accuracy_score(y_enc, proba.argmax(axis=1)),
            "log_loss": log_loss(y_enc, proba, labels=np.arange(n_classes)),
            "brier":    compute_brier(y_onehot, proba),
            "rps":      compute_rps(y_onehot, proba),
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "mode": self.mode,
                "calibrated_model": self.calibrated_model,
                "label_encoder": self.label_encoder,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.mode = data["mode"]
        self.calibrated_model = data["calibrated_model"]
        self.label_encoder = data["label_encoder"]


def train_and_log(
    features: pd.DataFrame,
    mode: str,
    n_trials: int = 100,
    experiment_name: str = "wc2026",
) -> WorldCupXGBModel:
    """
    Full train pipeline: filter data, tune, fit, evaluate on WC2022 holdout,
    log everything to MLflow. Returns fitted model.
    """
    train = features[features["date"] <= "2018-12-31"].dropna(subset=["outcome"])
    val = features[
        (features["tournament"] == "FIFA World Cup") &
        (features["date"] >= "2019-01-01") &
        (features["date"] <= "2022-12-31")
    ].dropna(subset=["outcome"])

    if mode == "binary":
        train = train[train["outcome"] != 0]
        val   = val[val["outcome"] != 0]

    X_train = train[FEATURE_COLS].fillna(0.0)
    y_train = train["outcome"].astype(int).values
    dates_train = train["date"]

    X_val = val[FEATURE_COLS].fillna(0.0)
    y_val = val["outcome"].astype(int).values

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"xgb_{mode}"):
        model = WorldCupXGBModel(mode=mode)

        print(f"Tuning {mode} model ({n_trials} trials)...")
        best_params = model.tune(X_train, y_train, dates_train, n_trials=n_trials)
        mlflow.log_params(best_params)

        print("Fitting calibrated model...")
        model.fit(X_train, y_train, dates_train, best_params)

        if len(X_val) > 0:
            metrics = model.evaluate(X_val, y_val)
            mlflow.log_metrics(metrics)
            print(f"WC2022 holdout — {metrics}")

    return model
