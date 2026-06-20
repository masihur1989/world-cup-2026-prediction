import os
import pandas as pd
import pytest
from src import pipeline

pytestmark = pytest.mark.skipif(
    not os.path.exists("data/processed/xgb_3class.pkl"),
    reason="requires trained model artifacts",
)


def test_stage_simulate_writes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROCESSED", tmp_path)
    import src.predictions as predmod
    monkeypatch.setattr(predmod, "PROCESSED", tmp_path)
    monkeypatch.setattr(predmod, "PREDICTIONS_DIR", tmp_path / "predictions")

    cfg = {"trials": 1, "simulations": 20, "as_of": "2026-06-10",
           "label": "pre_tournament", "force": True}
    pipeline.run_simulate_only(cfg)
    assert (tmp_path / "predictions" / "2026-06-10__pre_tournament.csv").exists()
    assert (tmp_path / "simulation_results.csv").exists()
