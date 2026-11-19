"""Post-hoc probability calibration (Platt Scaling & Isotonic Regression) on validation period."""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _to_logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


class ProbabilityCalibrator:
    """
    Calibrates raw classifier probabilities into well-calibrated annual M&A target probabilities
    using the temporal validation fold (Platt Scaling / Sigmoid vs. Isotonic Regression).
    """

    def __init__(self, method: str = "auto"):
        """
        Parameters
        ----------
        method : {'platt', 'isotonic', 'auto'}
            When 'auto', fits both Platt (sigmoid) and regularized Isotonic on validation data
            and selects the one achieving the lower Brier score while preserving ranking.
        """
        self.method = method.lower()
        self.base_model_: Any = None
        self.selected_method_: str = "platt"
        self.platt_lr_: LogisticRegression = LogisticRegression(C=1.0)
        self.isotonic_: IsotonicRegression = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.95)
        self.val_brier_scores_: Dict[str, float] = {}

    def _raw_probs(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base_model_.predict_proba(X)
        return np.asarray(raw[:, 1] if raw.ndim == 2 else raw, dtype=float)

    def fit(self, base_model: Any, X_val: pd.DataFrame, y_val: np.ndarray) -> "ProbabilityCalibrator":
        self.base_model_ = base_model
        y_arr = np.asarray(y_val, dtype=int)
        raw_p = self._raw_probs(X_val)
        logits = _to_logit(raw_p).reshape(-1, 1)

        # Fit Platt scaling on validation logits
        self.platt_lr_.fit(logits, y_arr)
        platt_p = self.platt_lr_.predict_proba(logits)[:, 1]
        platt_brier = float(brier_score_loss(y_arr, platt_p))

        # Fit Isotonic regression on raw validation probabilities
        self.isotonic_.fit(raw_p, y_arr)
        iso_p = self.isotonic_.transform(raw_p)
        iso_brier = float(brier_score_loss(y_arr, iso_p))

        self.val_brier_scores_ = {
            "uncalibrated": float(brier_score_loss(y_arr, raw_p)),
            "platt": platt_brier,
            "isotonic": iso_brier,
        }

        if self.method == "platt":
            self.selected_method_ = "platt"
        elif self.method == "isotonic":
            self.selected_method_ = "isotonic"
        else:
            # Prefer Platt scaling on small validation samples if isotonic creates excessive ties
            unique_iso = len(np.unique(np.round(iso_p, 4)))
            if unique_iso >= max(8, int(len(y_arr) * 0.05)) and iso_brier < platt_brier * 0.98:
                self.selected_method_ = "isotonic"
            else:
                self.selected_method_ = "platt"

        return self

    def predict_proba(self, X: pd.DataFrame, method_override: str = "") -> np.ndarray:
        raw_p = self._raw_probs(X)
        active_method = (method_override or self.selected_method_).lower()
        if active_method == "isotonic":
            cal_p = self.isotonic_.transform(raw_p)
        elif active_method == "uncalibrated":
            cal_p = raw_p
        else:
            logits = _to_logit(raw_p).reshape(-1, 1)
            cal_p = self.platt_lr_.predict_proba(logits)[:, 1]
        return np.clip(np.asarray(cal_p, dtype=float), 0.001, 0.990)

    def get_reliability_curve(
        self, X: pd.DataFrame, y_true: np.ndarray, n_bins: int = 8
    ) -> pd.DataFrame:
        y_arr = np.asarray(y_true, dtype=int)
        rows = []
        for label in ("uncalibrated", "platt", "isotonic"):
            probs = self.predict_proba(X, method_override=label)
            frac_pos, mean_pred = calibration_curve(y_arr, probs, n_bins=n_bins, strategy="quantile")
            for mp, fp in zip(mean_pred, frac_pos):
                rows.append(
                    {
                        "method": label,
                        "mean_predicted_prob": float(mp),
                        "observed_frequency": float(fp),
                    }
                )
        return pd.DataFrame(rows)
