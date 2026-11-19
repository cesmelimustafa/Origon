"""Unit tests for Task 1: Benchmark generator, TimeBasedSplitter, and leakage-free preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.config import SplitConfig
from src.data.benchmark_generator import generate_historical_ma_panel
from src.data.preprocessor import FinancialPreprocessor, TimeBasedSplitter


def test_generate_historical_ma_panel_structure_and_target_rate():
    df = generate_historical_ma_panel(
        n_companies=200,
        years=[2019, 2020, 2021, 2022, 2023, 2024],
        target_rate=0.055,
        random_state=42,
    )
    assert isinstance(df, pd.DataFrame)
    required_cols = {
        "ticker",
        "company_name",
        "sector",
        "fiscal_year",
        "total_assets",
        "total_equity",
        "total_debt",
        "cash_and_equivalents",
        "current_assets",
        "current_liabilities",
        "revenue",
        "revenue_t_minus_1",
        "revenue_t_minus_3",
        "ebitda",
        "ebit",
        "net_income",
        "interest_expense",
        "operating_cash_flow",
        "capex",
        "market_cap",
        "stock_return_12m",
        "stock_volatility_12m",
        "q1_ebitda_margin",
        "q2_ebitda_margin",
        "q3_ebitda_margin",
        "q4_ebitda_margin",
        "is_acquired",
    }
    assert required_cols.issubset(set(df.columns))
    # Target rate should be realistic M&A base rate (~3.5% to 7.5%)
    pos_rate = df["is_acquired"].mean()
    assert 0.035 <= pos_rate <= 0.075


def test_time_based_splitter_prevents_temporal_leakage():
    df = generate_historical_ma_panel(
        n_companies=120,
        years=[2020, 2021, 2022, 2023, 2024],
        target_rate=0.05,
        random_state=7,
    )
    cfg = SplitConfig(train_end_year=2022, val_year=2023, test_start_year=2024)
    splitter = TimeBasedSplitter(cfg)
    train_df, val_df, test_df = splitter.split(df)

    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    assert train_df["fiscal_year"].max() <= 2022
    assert set(val_df["fiscal_year"].unique()) == {2023}
    assert test_df["fiscal_year"].min() >= 2024

    # Strict temporal ordering: max(train.year) < min(val.year) < min(test.year)
    assert train_df["fiscal_year"].max() < val_df["fiscal_year"].min()
    assert val_df["fiscal_year"].max() < test_df["fiscal_year"].min()


def test_financial_preprocessor_winsorizes_and_imputes_without_future_leakage():
    train_df = pd.DataFrame(
        {
            "sector": ["Technology", "Technology", "Technology", "Healthcare", "Healthcare"],
            "fiscal_year": [2021, 2021, 2022, 2021, 2022],
            "ratio_a": [10.0, 12.0, 1000.0, 5.0, 7.0],  # 1000.0 is extreme outlier
            "ratio_b": [1.0, np.nan, 3.0, 2.0, np.nan],
        }
    )
    test_df = pd.DataFrame(
        {
            "sector": ["Technology", "Healthcare"],
            "fiscal_year": [2024, 2024],
            "ratio_a": [5000.0, -900.0],  # Future extreme outliers clipped to train bounds
            "ratio_b": [np.nan, np.nan],  # Imputed with train sector medians
        }
    )

    prep = FinancialPreprocessor(
        feature_cols=["ratio_a", "ratio_b"],
        sector_col="sector",
        winsor_limits=(0.10, 0.90),
    )
    train_clean = prep.fit_transform(train_df)
    test_clean = prep.transform(test_df)

    assert not train_clean[["ratio_a", "ratio_b"]].isna().any().any()
    assert not test_clean[["ratio_a", "ratio_b"]].isna().any().any()

    # Future outlier 5000.0 in test_df must be clipped to train upper bound (<= 1000.0)
    assert test_clean.loc[0, "ratio_a"] <= train_clean["ratio_a"].max()
    # Technology train median for ratio_b is median([1.0, 3.0]) == 2.0
    assert np.isclose(test_clean.loc[0, "ratio_b"], 2.0)
    # Healthcare train median for ratio_b is 2.0
    assert np.isclose(test_clean.loc[1, "ratio_b"], 2.0)
