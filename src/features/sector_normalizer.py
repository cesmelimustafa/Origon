"""Cross-sectional (Region, Sector, Fiscal Year) robust Z-score normalizer."""

from typing import Dict, List, Tuple
import pandas as pd

from src.config import ALL_RAW_FEATURES


class SectorTimeNormalizer:
    """
    Computes peer-relative Robust Z-Scores ((x - median) / IQR) within each
    (region, sector, fiscal_year) cross-section so companies in US, Europe, and Turkey (BIST)
    are normalized against their regional sector peers.
    """

    def __init__(
        self,
        feature_cols: List[str] = ALL_RAW_FEATURES,
        sector_col: str = "sector",
        year_col: str = "fiscal_year",
        region_col: str = "region",
    ):
        self.feature_cols = feature_cols
        self.sector_col = sector_col
        self.year_col = year_col
        self.region_col = region_col
        self.fallback_stats_: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.global_stats_: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "SectorTimeNormalizer":
        for col in self.feature_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            g_med = float(s.median()) if not s.empty else 0.0
            g_iqr = float(s.quantile(0.75) - s.quantile(0.25)) if not s.empty else 1.0
            if g_iqr <= 1e-6:
                g_iqr = max(float(s.std()) if not s.empty else 1.0, 1e-3)
            self.global_stats_[col] = (g_med, g_iqr)

            if self.sector_col in df.columns:
                for sec, grp in df.groupby(self.sector_col):
                    gs = pd.to_numeric(grp[col], errors="coerce").dropna()
                    if len(gs) >= 2:
                        med = float(gs.median())
                        iqr = float(gs.quantile(0.75) - gs.quantile(0.25))
                        if iqr <= 1e-6:
                            iqr = max(float(gs.std()), g_iqr, 1e-3)
                        self.fallback_stats_[(str(sec), col)] = (med, iqr)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        group_keys = (
            [self.region_col, self.sector_col, self.year_col]
            if self.region_col in out.columns
            else [self.sector_col, self.year_col]
        )
        z_dict = {}

        for col in self.feature_cols:
            g_med, g_iqr = self.global_stats_.get(col, (0.0, 1.0))

            def _norm_group(s: pd.Series, feat: str = col, gm: float = g_med, gi: float = g_iqr) -> pd.Series:
                if len(s) >= 3:
                    med = float(s.median())
                    iqr = float(s.quantile(0.75) - s.quantile(0.25))
                    if iqr <= 1e-6:
                        iqr = max(float(s.std()), gi, 1e-3)
                else:
                    sec_name = str(out.loc[s.index[0], self.sector_col]) if len(s) > 0 else ""
                    med, iqr = self.fallback_stats_.get((sec_name, feat), (gm, gi))
                return ((s - med) / max(iqr, 1e-4)).clip(-6.0, 6.0)

            z_series = out.groupby(group_keys)[col].transform(_norm_group)
            z_dict[f"{col}_sector_z"] = z_series.fillna(0.0)

        return pd.concat([out, pd.DataFrame(z_dict, index=out.index)], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
