"""Unit tests for Task 2: 5-category financial feature engineering and sector-time normalization."""

import numpy as np
import pandas as pd
import pytest

from src.config import ALL_RAW_FEATURES
from src.data.benchmark_generator import generate_historical_ma_panel
from src.features.financial_ratios import compute_financial_features
from src.features.sector_normalizer import SectorTimeNormalizer


def test_compute_financial_features_produces_all_22_ratios_and_handles_zero_denominators():
    raw_df = pd.DataFrame(
        [
            {
                "ticker": "CHEAP",
                "company_name": "Cheap Target Co",
                "sector": "Technology",
                "fiscal_year": 2024,
                "total_assets": 1000.0,
                "total_equity": 400.0,
                "total_debt": 200.0,
                "cash_and_equivalents": 100.0,
                "current_assets": 300.0,
                "current_liabilities": 150.0,
                "revenue": 800.0,
                "revenue_t_minus_1": 650.0,
                "revenue_t_minus_3": 500.0,
                "ebitda": 160.0,
                "ebit": 120.0,
                "net_income": 80.0,
                "interest_expense": 12.0,
                "operating_cash_flow": 140.0,
                "capex": 40.0,
                "market_cap": 900.0,
                "stock_return_12m": -0.18,
                "stock_volatility_12m": 0.34,
                "q1_ebitda_margin": 0.17,
                "q2_ebitda_margin": 0.19,
                "q3_ebitda_margin": 0.21,
                "q4_ebitda_margin": 0.23,
                "is_acquired": 1,
            },
            {
                "ticker": "EXP",
                "company_name": "Expensive Peer Co",
                "sector": "Technology",
                "fiscal_year": 2024,
                "total_assets": 2000.0,
                "total_equity": 0.0,  # zero equity edge case!
                "total_debt": 500.0,
                "cash_and_equivalents": 100.0,
                "current_assets": 400.0,
                "current_liabilities": 0.0,  # zero current liabilities edge case!
                "revenue": 1200.0,
                "revenue_t_minus_1": 1100.0,
                "revenue_t_minus_3": 950.0,
                "ebitda": 0.0,  # zero ebitda edge case!
                "ebit": 50.0,
                "net_income": 20.0,
                "interest_expense": 0.0,  # zero interest expense edge case!
                "operating_cash_flow": 90.0,
                "capex": 70.0,
                "market_cap": 3600.0,
                "stock_return_12m": 0.25,
                "stock_volatility_12m": 0.22,
                "q1_ebitda_margin": 0.20,
                "q2_ebitda_margin": 0.18,
                "q3_ebitda_margin": 0.16,
                "q4_ebitda_margin": 0.14,
                "is_acquired": 0,
            },
        ]
    )

    feat_df = compute_financial_features(raw_df)

    # All 22 required financial features must be present and finite
    for col in ALL_RAW_FEATURES:
        assert col in feat_df.columns, f"Missing engineered feature: {col}"
        assert np.isfinite(feat_df[col]).all(), f"Non-finite values in feature: {col}"

    # Verify exact math for CHEAP row:
    # EV = 900 + 200 - 100 = 1000; EV/EBITDA = 1000 / 160 = 6.25
    assert np.isclose(feat_df.loc[0, "ev_to_ebitda"], 6.25)
    # P/B = 900 / 400 = 2.25
    assert np.isclose(feat_df.loc[0, "price_to_book"], 2.25)
    # FCF / Assets = (140 - 40) / 1000 = 0.10
    assert np.isclose(feat_df.loc[0, "fcf_to_assets"], 0.10)
    # Net Debt / EBITDA = (200 - 100) / 160 = 0.625
    assert np.isclose(feat_df.loc[0, "net_debt_to_ebitda"], 0.625)
    # 4Q margin momentum: [0.17, 0.19, 0.21, 0.23] has positive slope +0.02 per quarter
    assert np.isclose(feat_df.loc[0, "margin_momentum_4q"], 0.02, atol=1e-4)
    # CHEAP is cheaper than EXP so its sector EV/EBITDA discount must be positive
    assert feat_df.loc[0, "sector_ev_ebitda_discount"] > 0.0


def test_sector_time_normalizer_adds_robust_z_scores():
    panel = generate_historical_ma_panel(n_companies=60, years=[2022, 2023], random_state=11)
    feat_df = compute_financial_features(panel)

    norm = SectorTimeNormalizer(feature_cols=ALL_RAW_FEATURES)
    norm_df = norm.fit_transform(feat_df)

    z_cols = [f"{c}_sector_z" for c in ALL_RAW_FEATURES]
    for zc in z_cols:
        assert zc in norm_df.columns
        assert np.isfinite(norm_df[zc]).all()


def test_multi_region_fx_conversion_and_region_specific_sector_discount():
    from src.data.fetcher import convert_row_currency_to_usd

    # A BIST company reported in TRY (e.g., 35,000M TRY at USD/TRY=35 -> 1,000M USD)
    try_row = {"total_assets": 35000.0, "market_cap": 17500.0, "currency": "TRY"}
    usd_conv = convert_row_currency_to_usd(try_row, fx_to_usd={"TRY": 1.0 / 35.0, "USD": 1.0})
    assert np.isclose(usd_conv["total_assets"], 1000.0, rtol=1e-3)
    assert np.isclose(usd_conv["market_cap"], 500.0, rtol=1e-3)

    # Verify region-aware sector discount prevents Country Risk Premium bias:
    # US Tech peer median EV/EBITDA is ~20x, while Turkey (BIST) Tech median is ~6x.
    # A BIST Tech company trading at 8x EV/EBITDA is EXPENSIVE relative to BIST peers (discount < 0),
    # even though 8x is lower than the US 20x multiple.
    cross_market_df = pd.DataFrame(
        [
            {
                "ticker": "US1",
                "company_name": "US Tech 1",
                "region": "US (NYSE/NASDAQ)",
                "currency": "USD",
                "sector": "Technology",
                "fiscal_year": 2024,
                "total_assets": 5000.0,
                "total_equity": 2500.0,
                "total_debt": 500.0,
                "cash_and_equivalents": 500.0,
                "current_assets": 1500.0,
                "current_liabilities": 600.0,
                "revenue": 3000.0,
                "revenue_t_minus_1": 2600.0,
                "revenue_t_minus_3": 2000.0,
                "ebitda": 500.0,
                "ebit": 400.0,
                "net_income": 300.0,
                "interest_expense": 25.0,
                "operating_cash_flow": 450.0,
                "capex": 100.0,
                "market_cap": 10000.0,  # EV/EBITDA = 20x
                "stock_return_12m": 0.15,
                "stock_volatility_12m": 0.25,
                "q1_ebitda_margin": 0.16,
                "q2_ebitda_margin": 0.16,
                "q3_ebitda_margin": 0.17,
                "q4_ebitda_margin": 0.17,
                "is_acquired": 0,
            },
            {
                "ticker": "US2",
                "company_name": "US Tech 2",
                "region": "US (NYSE/NASDAQ)",
                "currency": "USD",
                "sector": "Technology",
                "fiscal_year": 2024,
                "total_assets": 5000.0,
                "total_equity": 2500.0,
                "total_debt": 500.0,
                "cash_and_equivalents": 500.0,
                "current_assets": 1500.0,
                "current_liabilities": 600.0,
                "revenue": 3000.0,
                "revenue_t_minus_1": 2600.0,
                "revenue_t_minus_3": 2000.0,
                "ebitda": 500.0,
                "ebit": 400.0,
                "net_income": 300.0,
                "interest_expense": 25.0,
                "operating_cash_flow": 450.0,
                "capex": 100.0,
                "market_cap": 11000.0,  # EV/EBITDA = 22x
                "stock_return_12m": 0.18,
                "stock_volatility_12m": 0.25,
                "q1_ebitda_margin": 0.16,
                "q2_ebitda_margin": 0.16,
                "q3_ebitda_margin": 0.17,
                "q4_ebitda_margin": 0.17,
                "is_acquired": 0,
            },
            {
                "ticker": "TR_CHEAP.IS",
                "company_name": "BIST Cheap Tech",
                "region": "Turkey (Borsa İstanbul - BIST)",
                "currency": "TRY",
                "sector": "Technology",
                "fiscal_year": 2024,
                "total_assets": 400.0,
                "total_equity": 200.0,
                "total_debt": 50.0,
                "cash_and_equivalents": 50.0,
                "current_assets": 150.0,
                "current_liabilities": 60.0,
                "revenue": 300.0,
                "revenue_t_minus_1": 260.0,
                "revenue_t_minus_3": 200.0,
                "ebitda": 50.0,
                "ebit": 40.0,
                "net_income": 30.0,
                "interest_expense": 5.0,
                "operating_cash_flow": 45.0,
                "capex": 10.0,
                "market_cap": 250.0,  # EV/EBITDA = 5x
                "stock_return_12m": -0.10,
                "stock_volatility_12m": 0.38,
                "q1_ebitda_margin": 0.16,
                "q2_ebitda_margin": 0.16,
                "q3_ebitda_margin": 0.17,
                "q4_ebitda_margin": 0.17,
                "is_acquired": 1,
            },
            {
                "ticker": "TR_EXP.IS",
                "company_name": "BIST Expensive Tech",
                "region": "Turkey (Borsa İstanbul - BIST)",
                "currency": "TRY",
                "sector": "Technology",
                "fiscal_year": 2024,
                "total_assets": 400.0,
                "total_equity": 200.0,
                "total_debt": 50.0,
                "cash_and_equivalents": 50.0,
                "current_assets": 150.0,
                "current_liabilities": 60.0,
                "revenue": 300.0,
                "revenue_t_minus_1": 260.0,
                "revenue_t_minus_3": 200.0,
                "ebitda": 50.0,
                "ebit": 40.0,
                "net_income": 30.0,
                "interest_expense": 5.0,
                "operating_cash_flow": 45.0,
                "capex": 10.0,
                "market_cap": 450.0,  # EV/EBITDA = 9x (expensive for BIST, though < 20x US!)
                "stock_return_12m": 0.25,
                "stock_volatility_12m": 0.35,
                "q1_ebitda_margin": 0.16,
                "q2_ebitda_margin": 0.16,
                "q3_ebitda_margin": 0.17,
                "q4_ebitda_margin": 0.17,
                "is_acquired": 0,
            },
        ]
    )

    feat_cm = compute_financial_features(cross_market_df)
    # TR_EXP.IS has EV/EBITDA=9x vs BIST Tech median 7x -> must have negative discount (< 0)
    tr_exp_disc = feat_cm.loc[feat_cm["ticker"] == "TR_EXP.IS", "sector_ev_ebitda_discount"].iloc[0]
    tr_cheap_disc = feat_cm.loc[feat_cm["ticker"] == "TR_CHEAP.IS", "sector_ev_ebitda_discount"].iloc[0]
    assert tr_exp_disc < 0.0
    assert tr_cheap_disc > 0.0

