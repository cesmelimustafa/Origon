"""Unit tests for Task 3: Class imbalance handling, model training, probability calibration, and IB metrics."""

import numpy as np
import pandas as pd
import pytest

from src.config import ALL_RAW_FEATURES, SplitConfig
from src.data.benchmark_generator import generate_historical_ma_panel
from src.data.preprocessor import FinancialPreprocessor, TimeBasedSplitter
from src.features.financial_ratios import compute_financial_features
from src.models.calibration import ProbabilityCalibrator
from src.models.evaluator import compute_decile_lift_table, evaluate_ma_predictions
from src.models.imbalance import prepare_imbalance_strategy
from src.models.trainer import MAModelTrainer


def test_evaluate_ma_predictions_computes_exact_lift_and_precision_at_k():
    # 100 companies, 5 acquired (5% base rate)
    y_true = np.zeros(100, dtype=int)
    y_true[:4] = 1  # 4 of the top 10 highest-scored companies are true acquisitions
    y_true[50] = 1  # 1 outside top 10
    y_prob = np.linspace(0.90, 0.01, 100)

    metrics = evaluate_ma_predictions(y_true, y_prob, k_list=[10, 20])

    # Precision@10 = 4 / 10 = 0.40
    assert np.isclose(metrics["precision_at_10"], 0.40)
    # Top 10% is top 10 samples -> Precision@Top10% = 0.40; Base Rate = 0.05 -> Lift@Top10% = 0.40 / 0.05 = 8.0x
    assert np.isclose(metrics["precision_top_10pct"], 0.40)
    assert np.isclose(metrics["lift_top_10pct"], 8.0)
    # Recall@Top10% = 4 / 5 = 0.80
    assert np.isclose(metrics["recall_top_10pct"], 0.80)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.5 < metrics["roc_auc"] <= 1.0
    assert 0.3 < metrics["pr_auc"] <= 1.0

    decile_df = compute_decile_lift_table(y_true, y_prob)
    assert len(decile_df) == 10
    assert decile_df.loc[0, "decile"] == 1
    assert np.isclose(decile_df.loc[0, "lift"], 8.0)


def test_imbalance_strategies_and_calibrated_models_train_and_predict():
    panel = generate_historical_ma_panel(
        n_companies=90,
        years=[2020, 2021, 2022, 2023, 2024],
        target_rate=0.06,
        random_state=42,
    )
    feat_df = compute_financial_features(panel)
    splitter = TimeBasedSplitter(SplitConfig(train_end_year=2022, val_year=2023, test_start_year=2024))
    train_df, val_df, test_df = splitter.split(feat_df)

    prep = FinancialPreprocessor(feature_cols=ALL_RAW_FEATURES)
    train_clean = prep.fit_transform(train_df)
    val_clean = prep.transform(val_df)
    test_clean = prep.transform(test_df)

    X_train, y_train = train_clean[ALL_RAW_FEATURES], train_clean["is_acquired"].values
    X_val, y_val = val_clean[ALL_RAW_FEATURES], val_clean["is_acquired"].values
    X_test, y_test = test_clean[ALL_RAW_FEATURES], test_clean["is_acquired"].values

    # Check SMOTE only augments training rows
    X_sm, y_sm, spw = prepare_imbalance_strategy(X_train, y_train, strategy="smote", random_state=42)
    assert len(X_sm) > len(X_train)
    assert spw == 1.0

    # Train Logistic Regression, XGBoost, and LightGBM with scale_pos_weight
    trainer = MAModelTrainer(feature_cols=ALL_RAW_FEATURES, imbalance_strategy="scale_pos_weight", random_state=42)
    fitted_models = trainer.fit_all(X_train, y_train)
    assert {"logistic_regression", "xgboost", "lightgbm"} == set(fitted_models.keys())

    # Calibrate best model on validation set (Platt vs Isotonic)
    calibrator = ProbabilityCalibrator(method="auto")
    calibrator.fit(fitted_models["lightgbm"], X_val, y_val)
    cal_probs = calibrator.predict_proba(X_test)

    assert cal_probs.shape == (len(X_test),)
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))

    test_metrics = evaluate_ma_predictions(y_test, cal_probs)
    assert test_metrics["lift_top_10pct"] > 1.5
    assert test_metrics["roc_auc"] > 0.65
