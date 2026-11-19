"""Unit tests for Task 4: SHAP XAI engine extracting Top-3 bullish M&A drivers and Top-3 deterrents."""

import numpy as np
import pandas as pd
import pytest

from src.config import ALL_RAW_FEATURES
from src.data.benchmark_generator import generate_historical_ma_panel
from src.data.preprocessor import FinancialPreprocessor
from src.features.financial_ratios import compute_financial_features
from src.models.explainer import MAExplainer
from src.models.trainer import MAModelTrainer


def test_ma_explainer_extracts_top3_increasing_and_decreasing_factors_per_company():
    panel = generate_historical_ma_panel(n_companies=80, years=[2022, 2023], random_state=21)
    feat_df = compute_financial_features(panel)
    prep = FinancialPreprocessor(feature_cols=ALL_RAW_FEATURES)
    clean_df = prep.fit_transform(feat_df)

    X = clean_df[ALL_RAW_FEATURES]
    y = clean_df["is_acquired"].values

    trainer = MAModelTrainer(feature_cols=ALL_RAW_FEATURES, imbalance_strategy="scale_pos_weight", random_state=21)
    models = trainer.fit_all(X, y)

    explainer = MAExplainer(model=models["lightgbm"], feature_cols=ALL_RAW_FEATURES)
    shap_matrix = explainer.compute_shap_values(X.iloc[:15])

    assert shap_matrix.shape == (15, len(ALL_RAW_FEATURES))
    assert np.isfinite(shap_matrix).all()

    company_explanation = explainer.explain_company(clean_df.iloc[0], shap_row=shap_matrix[0])
    assert "top_positive_drivers" in company_explanation
    assert "top_negative_deterrents" in company_explanation
    assert "primary_rationale_tr" in company_explanation

    pos_drivers = company_explanation["top_positive_drivers"]
    neg_drivers = company_explanation["top_negative_deterrents"]

    assert len(pos_drivers) == 3
    assert len(neg_drivers) == 3
    # Sorted descending by positive SHAP impact
    assert pos_drivers[0]["shap_value"] >= pos_drivers[1]["shap_value"] >= pos_drivers[2]["shap_value"]
    # Sorted ascending (most negative first) for deterrents
    assert neg_drivers[0]["shap_value"] <= neg_drivers[1]["shap_value"] <= neg_drivers[2]["shap_value"]
    assert isinstance(company_explanation["primary_rationale_tr"], str)
    assert len(company_explanation["primary_rationale_tr"]) > 15

    # Also verify exact SHAP extraction for XGBoost (xgboost 3.x compatibility) and Logistic Regression
    for model_key in ("xgboost", "logistic_regression"):
        exp_other = MAExplainer(model=models[model_key], feature_cols=ALL_RAW_FEATURES, background_data=X.iloc[:30])
        mat_other = exp_other.compute_shap_values(X.iloc[:10])
        assert mat_other.shape == (10, len(ALL_RAW_FEATURES))
        assert np.isfinite(mat_other).all()

