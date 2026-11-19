"""Tests that pin the corporate finance literature properties of Modules 1 to 4 and the live data fetcher."""

import sys
import types

import numpy as np
import pandas as pd

from src.advisory.buyer_matcher import BuyerMatchingEngine
from src.advisory.lbo_simulator import LBORiskSimulator, deductible_interest
from src.advisory.premium_engine import (
    AcquisitionPremiumEngine,
    board_acceptance_probability,
    generate_precedent_transactions_dataset,
    reference_point_premium_pct,
)
from src.advisory.target_resolver import resolve_target_profile
from src.features.financial_ratios import compute_financial_features, palepu_growth_resource_mismatch


def _base_panel(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rev = 500.0 + 20.0 * i
        rows.append(
            {
                "ticker": f"T{i}",
                "region": "US (NYSE/NASDAQ)",
                "sector": "Industrials",
                "fiscal_year": 2024,
                "total_assets": 1000.0,
                "total_equity": 450.0,
                "total_debt": 250.0 + rng.normal(0, 30),
                "cash_and_equivalents": 80.0 + rng.normal(0, 10),
                "current_assets": 300.0,
                "current_liabilities": 180.0,
                "revenue": rev,
                "revenue_t_minus_1": rev / (1.05 + 0.01 * i),
                "revenue_t_minus_3": rev / 1.18,
                "ebitda": rev * 0.18,
                "ebit": rev * 0.14,
                "net_income": rev * 0.08,
                "interest_expense": 12.0,
                "operating_cash_flow": rev * 0.12,
                "capex": rev * 0.04,
                "market_cap": 900.0 + 15.0 * i,
                "stock_return_12m": 0.05,
                "stock_volatility_12m": 0.25,
                "q1_ebitda_margin": 0.17,
                "q2_ebitda_margin": 0.18,
                "q3_ebitda_margin": 0.18,
                "q4_ebitda_margin": 0.19,
            }
        )
    return pd.DataFrame(rows)


# Module 1


def test_palepu_mismatch_uses_growth_liquidity_and_leverage_jointly():
    df = pd.DataFrame(
        {
            "region": ["R"] * 6,
            "sector": ["S"] * 6,
            "fiscal_year": [2024] * 6,
            "rev_growth_1y": [0.05, 0.06, 0.07, 0.08, 0.40, 0.00],
            "cash_to_assets": [0.10, 0.11, 0.12, 0.10, 0.02, 0.40],
            "debt_to_equity": [0.80, 0.85, 0.90, 0.80, 2.50, 0.05],
        }
    )
    keys = ["region", "sector", "fiscal_year"]
    grm = palepu_growth_resource_mismatch(df, keys)
    balanced = grm.iloc[:4].max()
    # High growth, low liquidity, high leverage and low growth, high liquidity, low leverage are both mismatched.
    assert grm.iloc[4] > 2.0 * balanced
    assert grm.iloc[5] > 2.0 * balanced

    # Leverage must matter: same growth and liquidity, higher leverage increases the high growth mismatch.
    df_low_lev = df.copy()
    df_low_lev.loc[4, "debt_to_equity"] = 0.85
    grm_low = palepu_growth_resource_mismatch(df_low_lev, keys)
    assert grm.iloc[4] > grm_low.iloc[4]


def test_negative_ebitda_multiples_are_flagged_not_meaningful():
    panel = _base_panel()
    panel.loc[3, "ebitda"] = -15.0
    feats = compute_financial_features(panel)
    assert int(feats.loc[3, "ebitda_meaningful"]) == 0
    assert np.isnan(feats.loc[3, "ev_to_ebitda"])
    assert np.isnan(feats.loc[3, "sector_ev_ebitda_discount"])
    assert np.isnan(feats.loc[3, "net_debt_to_ebitda"])
    assert feats.loc[3, "ebitda_margin"] < 0.0
    # Peer medians ignore the not meaningful observation, so other firms keep finite discounts.
    assert feats.drop(index=3)["sector_ev_ebitda_discount"].notna().all()


def test_fetcher_ebitda_fallback_is_expressed_in_millions(monkeypatch):
    from src.data.fetcher import YFinanceDataFetcher

    info = {
        "marketCap": 2_000_000_000,
        "currency": "USD",
        "financialCurrency": "USD",
        "totalRevenue": 1_000_000_000,
        "ebitda": None,
        "ebitdaMargins": 0.12,
        "totalDebt": 300_000_000,
        "totalCash": 100_000_000,
        "totalAssets": 1_500_000_000,
        "currentPrice": 40.0,
        "fiftyTwoWeekHigh": 50.0,
        "shortName": "Test Corp",
    }

    class _FakeTicker:
        def __init__(self, sym):
            self.info = info

    fake_mod = types.ModuleType("yfinance")
    fake_mod.Ticker = _FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", fake_mod)

    row = YFinanceDataFetcher()._fetch_single_ticker("US (NYSE/NASDAQ)", "Industrials", "TST", 2024, {"USD": 1.0})
    assert row is not None
    assert np.isclose(row["revenue"], 1000.0, atol=0.5)
    assert np.isclose(row["ebitda"], 120.0, atol=0.5)
    assert np.isclose(row["market_cap"], 2000.0, atol=0.5)
    assert np.isclose(row["dist_from_52w_high"], -0.20, atol=1e-6)


# Module 2


def test_buyer_matching_reports_accretion_and_antitrust_friction():
    target = resolve_target_profile(
        custom_payload={
            "company_name": "Large Software Platform",
            "region": "US (NYSE/NASDAQ)",
            "sector": "Technology",
            "revenue": 9000.0,
            "ebitda": 2500.0,
            "total_debt": 1500.0,
            "cash_and_equivalents": 900.0,
            "market_cap": 30000.0,
            "rev_growth_1y": 0.14,
        }
    )
    res = BuyerMatchingEngine().match_buyers(target, top_k=5, lang="en")
    strategic = res["all_strategic"] if "all_strategic" in res else res["top_strategic"]
    for cand in strategic:
        assert "accretion_spread_pct" in cand
        assert "antitrust_penalty_pts" in cand
        assert cand["antitrust_penalty_pts"] >= 0.0
    penalties = [c["antitrust_penalty_pts"] for c in strategic if c["focus_sector"] in ("Technology",)]
    assert any(p > 0.0 for p in penalties)


# Module 3


def test_reference_point_premium_is_ratio_of_52w_high_to_price():
    # A stock 20 percent below its high needs a 25 percent premium to reach the high.
    assert np.isclose(reference_point_premium_pct(-0.20), 25.0)
    assert np.isclose(reference_point_premium_pct(-0.50), 100.0)
    assert np.isclose(reference_point_premium_pct(0.0), 0.0)


def test_board_acceptance_jumps_at_52w_high_reference_point():
    ref = reference_point_premium_pct(-0.20)
    below = board_acceptance_probability(ref - 0.01, inflection=30.0, spread=6.0, reference_premium=ref)
    above = board_acceptance_probability(ref + 0.01, inflection=30.0, spread=6.0, reference_premium=ref)
    assert above - below > 5.0
    grid = np.arange(0.0, 80.0, 0.5)
    probs = [board_acceptance_probability(p, 30.0, 6.0, ref) for p in grid]
    assert all(probs[i] <= probs[i + 1] + 1e-9 for i in range(len(probs) - 1))


def test_premium_quantiles_never_cross_and_conformal_band_covers_holdout():
    engine = AcquisitionPremiumEngine(random_state=42).fit(
        generate_precedent_transactions_dataset(n_transactions=640, random_state=42)
    )
    holdout = generate_precedent_transactions_dataset(n_transactions=400, random_state=7)
    lo, med, hi = engine.predict_premium_band(holdout)
    assert np.all(lo <= med + 1e-9)
    assert np.all(med <= hi + 1e-9)
    y = holdout["acquisition_premium_pct"].values
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    # Nominal 70 percent band from split conformal quantile regression.
    assert 0.62 <= coverage <= 0.88


# Module 4


def test_interest_deductibility_follows_regional_tax_regime():
    # US and EU: deductible interest capped at 30 percent of EBITDA.
    assert np.isclose(deductible_interest(50.0, 100.0, 600.0, 300.0, "US (NYSE/NASDAQ)"), 30.0)
    assert np.isclose(deductible_interest(20.0, 100.0, 600.0, 300.0, "Europe (Euronext/XETRA/LSE)"), 20.0)
    # Turkey: 10 percent of financing expense on the debt share exceeding equity is disallowed.
    # Excess share = (600 - 300) / 600 = 0.5, disallowed = 50 * 0.5 * 0.10 = 2.5
    assert np.isclose(deductible_interest(50.0, 100.0, 600.0, 300.0, "Turkey (Borsa İstanbul - BIST)"), 47.5)
    assert np.isclose(deductible_interest(50.0, 100.0, 200.0, 300.0, "Turkey (Borsa İstanbul - BIST)"), 50.0)


def test_lbo_schedule_uses_average_debt_interest_fees_and_covenants():
    target = resolve_target_profile(
        custom_payload={
            "company_name": "Nordic Industrial Systems",
            "region": "US (NYSE/NASDAQ)",
            "sector": "Industrials",
            "revenue": 1200.0,
            "ebitda": 240.0,
            "total_debt": 300.0,
            "cash_and_equivalents": 110.0,
            "market_cap": 1950.0,
            "rev_growth_1y": 0.06,
        }
    )
    sim = LBORiskSimulator(n_iterations=2000, random_state=1)
    res = sim.simulate(
        target=target,
        entry_ev_usd_m=2400.0,
        total_debt_ebitda_turns=6.0,
        senior_debt_share=0.70,
        senior_interest_rate=0.08,
        subordinated_interest_rate=0.12,
        exit_ev_ebitda_multiple=10.0,
        min_cash_buffer_pct_rev=0.04,
    )
    cs = res["capital_structure"]
    # Sources and uses: equity funds EV plus fees plus opening cash less debt.
    expected_equity = (
        2400.0 + cs["transaction_fees_usd_m"] + cs["initial_min_cash_buffer_usd_m"] - cs["total_debt_usd_m"]
    )
    assert np.isclose(cs["sponsor_equity_usd_m"], expected_equity, atol=0.5)
    assert cs["transaction_fees_usd_m"] > 0.0

    sched = res["projection_schedule"]
    sen_rate, sub_rate = 0.08, 0.12
    prev_sen, prev_sub = cs["senior_debt_usd_m"], cs["subordinated_debt_usd_m"]
    for row in sched:
        avg_sen = 0.5 * (prev_sen + row["ending_senior_debt_usd_m"])
        avg_sub = 0.5 * (prev_sub + row["ending_sub_debt_usd_m"])
        assert np.isclose(row["interest_expense_usd_m"], avg_sen * sen_rate + avg_sub * sub_rate, rtol=0.02, atol=0.5)
        assert row["deductible_interest_usd_m"] <= 0.30 * row["ebitda_usd_m"] + 0.5
        assert row["deductible_interest_usd_m"] <= row["interest_expense_usd_m"] + 0.5
        assert "interest_coverage" in row
        prev_sen, prev_sub = row["ending_senior_debt_usd_m"], row["ending_sub_debt_usd_m"]

    summ = res["summary"]
    assert 0.0 <= summ["covenant_breach_probability_pct"] <= 100.0
