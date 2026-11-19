"""Time-based train/val/test splitting and leakage-free financial preprocessing."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import DEFAULT_SPLIT_CONFIG, SplitConfig


class TimeBasedSplitter:
    """Strictly temporal train / validation / out-of-time test splitter to prevent look-ahead bias."""

    def __init__(self, config: SplitConfig = DEFAULT_SPLIT_CONFIG):
        self.config = config

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        year_col = self.config.year_col
        if year_col not in df.columns:
            raise ValueError(f"Column '{year_col}' is required for TimeBasedSplitter.")

        train_df = df[df[year_col] <= self.config.train_end_year].copy().reset_index(drop=True)
        val_df = df[df[year_col] == self.config.val_year].copy().reset_index(drop=True)
        test_df = df[df[year_col] >= self.config.test_start_year].copy().reset_index(drop=True)

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                f"One or more temporal splits are empty: "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )

        return train_df, val_df, test_df


class FinancialPreprocessor:
    """
    Leakage-free winsorization and sector-median imputation.

    All quantile clipping bounds and sector median fallback values are fit strictly
    on the training split and applied unchanged to validation and future test splits.
    """

    def __init__(
        self,
        feature_cols: List[str],
        sector_col: str = "sector",
        winsor_limits: Tuple[float, float] = (0.01, 0.99),
    ):
        self.feature_cols = feature_cols
        self.sector_col = sector_col
        self.winsor_limits = winsor_limits
        self.lower_bounds_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}
        self.sector_medians_: Dict[str, Dict[str, float]] = {}
        self.global_medians_: Dict[str, float] = {}
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FinancialPreprocessor":
        low_q, high_q = self.winsor_limits
        for col in self.feature_cols:
            series = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if valid.empty:
                self.lower_bounds_[col] = -1e6
                self.upper_bounds_[col] = 1e6
                self.global_medians_[col] = 0.0
            else:
                self.lower_bounds_[col] = float(valid.quantile(low_q))
                self.upper_bounds_[col] = float(valid.quantile(high_q))
                clipped = series.clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
                self.global_medians_[col] = float(clipped.median())

            # Sector-specific medians on clipped training values
            clipped_series = series.clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
            sec_meds = (
                clipped_series.groupby(train_df[self.sector_col]).median().dropna().to_dict()
                if self.sector_col in train_df.columns
                else {}
            )
            self.sector_medians_[col] = {str(k): float(v) for k, v in sec_meds.items()}

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FinancialPreprocessor must be fitted on training data before transform.")

        out = df.copy()
        for col in self.feature_cols:
            series = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if self.sector_col in out.columns and self.sector_medians_.get(col):
                sector_fill = out[self.sector_col].map(self.sector_medians_[col])
                series = series.fillna(sector_fill)
            series = series.fillna(self.global_medians_[col])
            series = series.clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
            out[col] = series.astype(float)

        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)
