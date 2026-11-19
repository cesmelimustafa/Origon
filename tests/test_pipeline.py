"""Unit and integration test for Task 5: End-to-end M&A prediction pipeline and artifact persistence."""

from pathlib import Path
import pandas as pd
import pytest

from src.pipeline import run_ma_pipeline


def test_run_ma_pipeline_executes_end_to_end_and_saves_artifacts(tmp_path: Path):
    results = run_ma_pipeline(
        n_companies=60,
        output_dir=tmp_path,
        use_live_yfinance=False,
        random_state=42,
    )

    assert "model_comparison_df" in results
    assert "ranked_targets_df" in results
    assert "decile_lift_df" in results
    assert "reliability_df" in results
    assert "champion_model_name" in results
    assert "champion_metrics" in results

    comp_df = results["model_comparison_df"]
    assert len(comp_df) >= 4  # Multiple models & imbalance strategies compared
    assert {"model_name", "imbalance_strategy", "pr_auc", "roc_auc", "brier_score", "lift_top_10pct"}.issubset(
        set(comp_df.columns)
    )

    ranked_df = results["ranked_targets_df"]
    assert "predicted_ma_prob" in ranked_df.columns
    assert "top_driver_1" in ranked_df.columns
    assert "top_deterrent_1" in ranked_df.columns
    # Sorted descending by predicted M&A probability
    assert ranked_df["predicted_ma_prob"].is_monotonic_decreasing

    # Verify saved artifacts on disk
    assert (tmp_path / "ranked_targets.parquet").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "decile_lift.csv").exists()
    assert (tmp_path / "pipeline_bundle.joblib").exists()


def test_api_schema_helpers_and_clean_text_formatting():
    from src.api.schemas import assign_clean_size_bucket, sanitize_display_text

    assert assign_clean_size_bucket(450.0) == "Küçük Ölçek"
    assert assign_clean_size_bucket(3500.0) == "Orta Ölçek"
    assert assign_clean_size_bucket(25000.0) == "Büyük Ölçek"
    assert "(" not in sanitize_display_text("PD / DD (P/B)")
    assert '"' not in sanitize_display_text('"Özel Vurgu"')


