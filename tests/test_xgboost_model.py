import numpy as np
import pandas as pd
import pytest
from src.xgboost_model import (
    WorldCupXGBModel,
    FEATURE_COLS,
    compute_rps,
    compute_brier,
)

@pytest.fixture
def small_Xy():
    """200 rows of synthetic features + 3-class targets."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = rng.choice([1, 0, -1], n)
    dates = pd.date_range("2000-01-01", periods=n, freq="7D")
    return X, y, dates

def test_feature_cols_length():
    assert len(FEATURE_COLS) == 28

def test_fit_3class(small_Xy):
    X, y, dates = small_Xy
    model = WorldCupXGBModel(mode="3class")
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50,
              "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1}
    model.fit(X, y, dates, params)
    assert model.calibrated_model is not None

def test_predict_proba_sums_to_1(small_Xy):
    X, y, dates = small_Xy
    model = WorldCupXGBModel(mode="3class")
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50,
              "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1}
    model.fit(X, y, dates, params)
    proba = model.predict_proba(X.iloc[:5])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

def test_fit_binary_no_draw_class(small_Xy):
    X, y, dates = small_Xy
    mask = y != 0
    X_ko, y_ko, dates_ko = X[mask], y[mask], dates[mask]
    model = WorldCupXGBModel(mode="binary")
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50,
              "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1}
    model.fit(X_ko, y_ko, dates_ko, params)
    proba = model.predict_proba(X_ko.iloc[:5])
    assert proba.shape[1] == 2
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

def test_compute_rps():
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert compute_rps(y_true, y_pred) == pytest.approx(0.0)

def test_compute_brier():
    y_true = np.array([[1, 0, 0]])
    y_pred = np.array([[1.0, 0.0, 0.0]])
    assert compute_brier(y_true, y_pred) == pytest.approx(0.0)

def test_compute_shap_shape(small_Xy):
    X, y, dates = small_Xy
    model = WorldCupXGBModel(mode="3class")
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50,
              "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1}
    model.fit(X, y, dates, params)
    shap_vals = model.compute_shap(X.iloc[:10])
    assert shap_vals is not None

def test_save_load_model(tmp_path, small_Xy):
    X, y, dates = small_Xy
    model = WorldCupXGBModel(mode="3class")
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50,
              "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1}
    model.fit(X, y, dates, params)
    path = str(tmp_path / "model.pkl")
    model.save(path)
    model2 = WorldCupXGBModel(mode="3class")
    model2.load(path)
    p1 = model.predict_proba(X.iloc[:3])
    p2 = model2.predict_proba(X.iloc[:3])
    np.testing.assert_allclose(p1, p2, atol=1e-5)
