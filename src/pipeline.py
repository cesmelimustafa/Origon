"""End-to-end M&A Target Prediction training, calibration, evaluation, and SHAP XAI orchestrator."""

from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import joblib
import numpy as np
import pandas as pd

from src.config import ALL_RAW_FEATURES, ARTIFACTS_DIR, DEFAULT_SPLIT_CONFIG
from src.data.fetcher import YFinanceDataFetcher
from src.data.preprocessor import FinancialPreprocessor, TimeBasedSplitter
from src.features.financial_ratios import compute_financial_features
from src.features.sector_normalizer import SectorTimeNormalizer
from src.models.calibration import ProbabilityCalibrator
from src.models.evaluator import compute_decile_lift_table, evaluate_ma_predictions
from src.models.explainer import MAExplainer
from src.models.trainer import MAModelTrainer


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression (ElasticNet L1/L2)",
    "xgboost": "XGBoost Classifier",
    "lightgbm": "LightGBM Classifier",
}


def run_ma_pipeline(
    n_companies: int = 420,
    output_dir: Optional[Path] = None,
    use_live_yfinance: bool = False,
    force_rebuild_data: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute the complete M&A Target Prediction pipeline:
    1. Panel data loading / generation
    2. 5-category corporate finance feature engineering
    3. Leakage-free temporal split (Train <= 2022, Val == 2023, Out-of-Time Test >= 2024)
    4. Winsorization, sector-median imputation, and peer robust Z-score normalization
    5. Model training across imbalance strategies (scale_pos_weight vs. SMOTE)
    6. Probability calibration (Platt Scaling & Isotonic Regression) on validation period
    7. Out-of-time evaluation (PR-AUC, ROC-AUC, Brier Score, Precision@K, Lift@Top10%)
    8. SHAP XAI deal rationale extraction (Top-3 Positive Drivers & Top-3 Deterrents per target)
    """
    out_dir = Path(output_dir) if output_dir is not None else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fetcher = YFinanceDataFetcher(cache_path=out_dir / "market_panel_cache.parquet")
    panel_df = fetcher.load_or_build_panel(
        n_companies=n_companies,
        use_live_yfinance=use_live_yfinance,
        force_rebuild=force_rebuild_data,
        random_state=random_state,
    )

    feat_df = compute_financial_features(panel_df)

    splitter = TimeBasedSplitter(DEFAULT_SPLIT_CONFIG)
    train_raw, val_raw, test_raw = splitter.split(feat_df)

    preprocessor = FinancialPreprocessor(
        feature_cols=ALL_RAW_FEATURES,
        sector_col=DEFAULT_SPLIT_CONFIG.sector_col,
        winsor_limits=DEFAULT_SPLIT_CONFIG.winsor_limits,
    )
    train_clean = preprocessor.fit_transform(train_raw)
    val_clean = preprocessor.transform(val_raw)
    test_clean = preprocessor.transform(test_raw)

    normalizer = SectorTimeNormalizer(feature_cols=ALL_RAW_FEATURES)
    train_norm = normalizer.fit_transform(train_clean)
    val_norm = normalizer.transform(val_clean)
    test_norm = normalizer.transform(test_clean)

    X_train, y_train = train_norm[ALL_RAW_FEATURES], train_norm["is_acquired"].values
    X_val, y_val = val_norm[ALL_RAW_FEATURES], val_norm["is_acquired"].values
    X_test, y_test = test_norm[ALL_RAW_FEATURES], test_norm["is_acquired"].values

    comparison_rows = []
    trained_registry: Dict[str, Dict[str, Any]] = {}

    for strategy in ("scale_pos_weight", "smote"):
        trainer = MAModelTrainer(
            feature_cols=ALL_RAW_FEATURES,
            imbalance_strategy=strategy,
            random_state=random_state,
        )
        models = trainer.fit_all(X_train, y_train)

        for model_key, model_obj in models.items():
            calibrator = ProbabilityCalibrator(method="auto")
            calibrator.fit(model_obj, X_val, y_val)
            test_probs = calibrator.predict_proba(X_test)
            metrics = evaluate_ma_predictions(y_test, test_probs, k_list=[10, 25, 50])

            reg_key = f"{model_key}__{strategy}"
            trained_registry[reg_key] = {
                "model_key": model_key,
                "display_name": f"{MODEL_DISPLAY_NAMES.get(model_key, model_key)} [{strategy}]",
                "imbalance_strategy": strategy,
                "model": model_obj,
                "calibrator": calibrator,
                "test_probs": test_probs,
                "metrics": metrics,
            }

            comparison_rows.append(
                {
                    "registry_key": reg_key,
                    "model_name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
                    "imbalance_strategy": strategy,
                    "calibration_method": calibrator.selected_method_,
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "brier_score": metrics["brier_score"],
                    "lift_top_10pct": metrics["lift_top_10pct"],
                    "precision_top_10pct": metrics["precision_top_10pct"],
                    "recall_top_10pct": metrics["recall_top_10pct"],
                    "precision_at_10": metrics["precision_at_10"],
                    "precision_at_25": metrics["precision_at_25"],
                }
            )

    model_comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(by=["pr_auc", "lift_top_10pct"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # Select best tree-based or overall model by PR-AUC & Lift@Top10%
    champion_key = str(model_comparison_df.loc[0, "registry_key"])
    champion_entry = trained_registry[champion_key]
    champion_model = champion_entry["model"]
    champion_calibrator = champion_entry["calibrator"]
    champion_probs = champion_entry["test_probs"]
    champion_metrics = champion_entry["metrics"]

    # SHAP XAI explanations on the out-of-time test universe
    explainer = MAExplainer(
        model=champion_model,
        feature_cols=ALL_RAW_FEATURES,
        background_data=X_train.iloc[: min(150, len(X_train))],
    )
    shap_matrix = explainer.compute_shap_values(X_test)

    scored_df = test_norm.copy()
    scored_df["predicted_ma_prob"] = np.round(champion_probs, 4)
    scored_df["predicted_ma_pct"] = np.round(champion_probs * 100.0, 2)
    annotated_df = explainer.build_explanations_table(scored_df, shap_matrix)

    # Sort descending by predicted M&A probability and keep corresponding SHAP matrix aligned
    sort_order = np.argsort(-annotated_df["predicted_ma_prob"].values)
    ranked_targets_df = annotated_df.iloc[sort_order].reset_index(drop=True)
    ranked_targets_df["ma_rank"] = np.arange(1, len(ranked_targets_df) + 1)
    sorted_shap_matrix = shap_matrix[sort_order]

    # Sector medians on the test cross-section (grouped by region & sector) for peer benchmarking cards
    med_group_cols = ["region", "sector"] if "region" in ranked_targets_df.columns else ["sector"]
    sector_medians_df = (
        ranked_targets_df.groupby(med_group_cols)[ALL_RAW_FEATURES].median().reset_index()
    )

    decile_lift_df = compute_decile_lift_table(y_test, champion_probs)
    reliability_df = champion_calibrator.get_reliability_curve(X_test, y_test, n_bins=8)

    # Save artifacts for Streamlit dashboard and CLI consumers
    ranked_targets_df.to_parquet(out_dir / "ranked_targets.parquet", index=False)
    model_comparison_df.to_csv(out_dir / "model_comparison.csv", index=False)
    decile_lift_df.to_csv(out_dir / "decile_lift.csv", index=False)
    reliability_df.to_csv(out_dir / "reliability_curve.csv", index=False)
    sector_medians_df.to_csv(out_dir / "sector_medians.csv", index=False)

    bundle = {
        "champion_key": champion_key,
        "champion_model_name": champion_entry["display_name"],
        "champion_metrics": champion_metrics,
        "feature_cols": ALL_RAW_FEATURES,
        "sorted_shap_matrix": sorted_shap_matrix,
        "preprocessor": preprocessor,
        "normalizer": normalizer,
        "champion_model": champion_model,
        "champion_calibrator": champion_calibrator,
    }
    joblib.dump(bundle, out_dir / "pipeline_bundle.joblib")

    return {
        "champion_key": champion_key,
        "champion_model_name": champion_entry["display_name"],
        "champion_metrics": champion_metrics,
        "model_comparison_df": model_comparison_df,
        "ranked_targets_df": ranked_targets_df,
        "decile_lift_df": decile_lift_df,
        "reliability_df": reliability_df,
        "sector_medians_df": sector_medians_df,
        "sorted_shap_matrix": sorted_shap_matrix,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the M&A Target Prediction pipeline.")
    parser.add_argument("--n-companies", type=int, default=420, help="Number of companies in universe.")
    parser.add_argument(
        "--no-live-yfinance",
        action="store_true",
        help="Disable live Yahoo Finance snapshot fetching (offline mode).",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild dataset cache from scratch.")
    args = parser.parse_args()

    res = run_ma_pipeline(
        n_companies=args.n_companies,
        use_live_yfinance=not args.no_live_yfinance,
        force_rebuild_data=args.force_rebuild,
    )
    print("=== M&A Target Prediction Pipeline Completed ===")
    print(f"Champion Model: {res['champion_model_name']}")
    print("Out-of-Time Metrics:", res["champion_metrics"])
    print("\nModel Comparison Table:")
    print(res["model_comparison_df"].to_string(index=False))
