"""Model training for M&A Target Prediction: Logistic Regression (L1/L2), XGBoost, and LightGBM."""

import os
from typing import Any, Dict, List

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.models.imbalance import prepare_imbalance_strategy


class MAModelTrainer:
    """Trains baseline penalized Logistic Regression, XGBoost, and LightGBM under a specified imbalance strategy."""

    def __init__(
        self,
        feature_cols: List[str],
        imbalance_strategy: str = "scale_pos_weight",
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.imbalance_strategy = imbalance_strategy
        self.random_state = random_state
        self.models_: Dict[str, Any] = {}
        self.scale_pos_weight_: float = 1.0

    def _build_models(self, pos_weight: float) -> Dict[str, Any]:
        lr_class_weight = (
            {0: 1.0, 1: pos_weight} if self.imbalance_strategy in ("scale_pos_weight", "class_weight") else None
        )
        logistic_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        l1_ratio=0.5,
                        C=0.5,
                        solver="saga",
                        max_iter=4000,
                        tol=1e-3,
                        class_weight=lr_class_weight,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        xgb_model = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=5,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=1,
        )

        lgbm_model = LGBMClassifier(
            n_estimators=180,
            num_leaves=15,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_samples=20,
            scale_pos_weight=pos_weight,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=1,
        )

        return {
            "logistic_regression": logistic_pipe,
            "xgboost": xgb_model,
            "lightgbm": lgbm_model,
        }

    def fit_all(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Any]:
        X_sub = X_train[self.feature_cols]
        X_res, y_res, pos_weight = prepare_imbalance_strategy(
            X_sub, y_train, strategy=self.imbalance_strategy, random_state=self.random_state
        )
        self.scale_pos_weight_ = pos_weight

        models = self._build_models(pos_weight)
        for name, model in models.items():
            model.fit(X_res, y_res)
            self.models_[name] = model

        return self.models_
