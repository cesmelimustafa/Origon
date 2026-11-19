"""Unit and API integration tests for Modules 2, 3, and 4 of the Origon M&A Advisory Decision Support System."""

import numpy as np
import pandas as pd

from src.advisory.buyer_catalog import build_pe_sponsor_catalog, build_strategic_buyer_catalog
from src.advisory.buyer_matcher import BuyerMatchingEngine
from src.advisory.lbo_simulator import LBORiskSimulator
from src.advisory.premium_engine import AcquisitionPremiumEngine, generate_precedent_transactions_dataset
from src.advisory.target_resolver import resolve_target_profile
from src.api.server import create_app
from tests.test_api import _invoke_asgi


def test_target_resolver_and_buyer_matching_engine_with_40_pe_sponsors():
    pe_df = build_pe_sponsor_catalog()
    strat_df = build_strategic_buyer_catalog()
    assert len(pe_df) == 40
    assert len(strat_df) >= 24

    # Custom private company target
    custom_target = resolve_target_profile(
        custom_payload={
            "company_name": "Anadolu Endüstriyel Otomasyon",
            "region": "Turkey (Borsa İstanbul - BIST)",
            "sector": "Technology",
            "revenue": 420.0,
            "ebitda": 92.0,
            "total_debt": 65.0,
            "cash_and_equivalents": 48.0,
            "market_cap": 680.0,
            "rev_growth_1y": 0.22,
            "stock_volatility_90d": 0.31,
            "dist_from_52w_high": -0.18,
        }
    )
    assert custom_target["is_custom"] is True
    assert custom_target["enterprise_value"] > 0

    matcher = BuyerMatchingEngine()
    res_tr = matcher.match_buyers(custom_target, top_k=5, lang="tr")
    res_en = matcher.match_buyers(custom_target, top_k=5, lang="en")

    assert len(res_tr["top_overall"]) == 5
    assert len(res_tr["top_strategic"]) == 5
    assert len(res_tr["top_financial"]) == 5

    top1 = res_tr["top_overall"][0]
    assert 0.0 <= top1["total_score"] <= 100.0
    assert {"nlp_fit_score", "financial_capacity_score", "deal_propensity_score", "synergy_score"}.issubset(
        set(top1["breakdown"].keys())
    )
    # Strict content rule: no parentheses or quotation marks in rationale
    assert "(" not in top1["rationale"] and ")" not in top1["rationale"]
    assert top1["rationale"] != res_en["top_overall"][0]["rationale"]


def test_acquisition_premium_engine_quantile_bands_and_board_acceptance_curve():
    prec_df = generate_precedent_transactions_dataset(n_transactions=480, random_state=42)
    assert len(prec_df) == 480
    assert "acquisition_premium_pct" in prec_df.columns

    engine = AcquisitionPremiumEngine(random_state=42)
    engine.fit(prec_df)

    target = resolve_target_profile(
        custom_payload={
            "company_name": "MedTech Diagnostics",
            "region": "Europe (Euronext/XETRA/LSE)",
            "sector": "Healthcare",
            "revenue": 850.0,
            "ebitda": 185.0,
            "total_debt": 210.0,
            "cash_and_equivalents": 95.0,
            "market_cap": 1650.0,
            "rev_growth_1y": 0.16,
            "stock_volatility_90d": 0.28,
            "dist_from_52w_high": -0.24,
        }
    )

    out = engine.recommend_offer(target, lang="tr")
    assert out["min_premium_pct"] < out["median_premium_pct"] < out["max_premium_pct"]
    assert out["offer_equity_min_usd_m"] < out["offer_equity_median_usd_m"] < out["offer_equity_max_usd_m"]

    # Verify Equity-to-EV bridge: Offer EV = Offer Equity + Debt (210) - Cash (95)
    expected_ev_med = out["offer_equity_median_usd_m"] + 210.0 - 95.0
    assert np.isclose(out["offer_ev_median_usd_m"], expected_ev_med, atol=0.5)

    # Board acceptance curve must be monotonically non-decreasing across premiums
    curve = out["acceptance_curve"]
    assert len(curve) >= 15
    probs = [pt["acceptance_probability_pct"] for pt in curve]
    assert all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1))
    assert len(out["precedent_deals"]) >= 5
    assert len(out["premium_drivers"]) >= 5


def test_lbo_simulator_enforces_minimum_cash_buffer_and_runs_5000_monte_carlo():
    target = resolve_target_profile(
        custom_payload={
            "company_name": "Nordic Industrial Systems",
            "region": "Europe (Euronext/XETRA/LSE)",
            "sector": "Industrials",
            "revenue": 1200.0,
            "ebitda": 240.0,
            "total_debt": 300.0,
            "cash_and_equivalents": 110.0,
            "market_cap": 1950.0,
            "rev_growth_1y": 0.09,
        }
    )

    sim = LBORiskSimulator(n_iterations=5000, random_state=42)
    res = sim.simulate(
        target=target,
        entry_ev_usd_m=2400.0,
        total_debt_ebitda_turns=4.5,
        senior_debt_share=0.70,
        senior_interest_rate=0.065,
        subordinated_interest_rate=0.105,
        exit_ev_ebitda_multiple=10.5,
        min_cash_buffer_pct_rev=0.04,
        lang="tr",
    )

    assert res["n_iterations"] == 5000
    assert "expected_irr_pct" in res["summary"]
    assert "median_moic" in res["summary"]
    assert 0.0 <= res["summary"]["stress_default_probability_pct"] <= 100.0

    # Verify Minimum Operating Cash Buffer Rule across all 5 projection years:
    # Ending cash must never be swept to zero; it stays >= required minimum cash buffer
    schedule = res["projection_schedule"]
    assert len(schedule) == 5
    for yr_row in schedule:
        assert yr_row["ending_cash_usd_m"] >= yr_row["min_cash_buffer_usd_m"] - 1e-2

    # Verify 5x5 Sensitivity Matrix
    matrix = res["sensitivity_matrix"]
    assert len(matrix["leverage_axes"]) == 5
    assert len(matrix["exit_multiple_axes"]) == 5
    assert len(matrix["cells"]) == 25
    assert len(res["irr_histogram"]) >= 16


def test_advisory_api_endpoints_support_public_and_custom_targets_in_tr_and_en():
    app = create_app()

    code_b, buyers_res = _invoke_asgi(
        app,
        "/api/v1/advisory/buyers?lang=tr",
        method="POST",
        body=b'{"ticker": "MPARK.IS"}',
    )
    assert code_b == 200
    assert len(buyers_res["top_overall"]) == 5

    code_p, prem_res = _invoke_asgi(
        app,
        "/api/v1/advisory/premium?lang=en",
        method="POST",
        body=b'{"ticker": "MPARK.IS"}',
    )
    assert code_p == 200
    assert prem_res["min_premium_pct"] < prem_res["median_premium_pct"] < prem_res["max_premium_pct"]

    code_l, lbo_res = _invoke_asgi(
        app,
        "/api/v1/advisory/lbo?lang=tr",
        method="POST",
        body=b'{"ticker": "MPARK.IS", "total_debt_ebitda_turns": 4.2, "exit_ev_ebitda_multiple": 11.0}',
    )
    assert code_l == 200
    assert lbo_res["n_iterations"] == 5000
    assert len(lbo_res["sensitivity_matrix"]["cells"]) == 25
