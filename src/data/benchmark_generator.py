"""Realistic multi-region historical M&A panel dataset generator (US, Europe, Turkey BIST)."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import MARKET_TICKERS_BY_REGION_AND_SECTOR, REGIONS, SECTORS


SECTOR_PROFILES: Dict[str, Dict[str, float]] = {
    "Technology": {"ev_ebitda": 18.0, "pb": 5.2, "margin": 0.24, "growth": 0.14, "hhi": 0.18},
    "Healthcare": {"ev_ebitda": 14.5, "pb": 3.8, "margin": 0.21, "growth": 0.11, "hhi": 0.15},
    "Industrials": {"ev_ebitda": 11.5, "pb": 2.9, "margin": 0.16, "growth": 0.07, "hhi": 0.12},
    "Consumer Discretionary": {"ev_ebitda": 12.0, "pb": 3.2, "margin": 0.14, "growth": 0.08, "hhi": 0.14},
    "Consumer Staples": {"ev_ebitda": 13.0, "pb": 3.5, "margin": 0.17, "growth": 0.05, "hhi": 0.20},
    "Energy": {"ev_ebitda": 7.5, "pb": 1.7, "margin": 0.26, "growth": 0.06, "hhi": 0.22},
    "Financials": {"ev_ebitda": 9.5, "pb": 1.5, "margin": 0.28, "growth": 0.06, "hhi": 0.16},
    "Materials": {"ev_ebitda": 9.0, "pb": 2.1, "margin": 0.18, "growth": 0.06, "hhi": 0.19},
}

REGION_PARAMETERS: Dict[str, Dict[str, object]] = {
    "US (NYSE/NASDAQ)": {
        "currency": "USD",
        "multiple_scale": 1.00,
        "base_log_assets": 8.0,
        "vol_addon": 0.00,
    },
    "Europe (Euronext/XETRA/LSE)": {
        "currency": "EUR",
        "multiple_scale": 0.80,  # European valuation discount vs US
        "base_log_assets": 7.7,
        "vol_addon": 0.02,
    },
    "Turkey (Borsa İstanbul - BIST)": {
        "currency": "TRY",
        "multiple_scale": 0.52,  # Higher Country Risk Premium (ERP) -> lower EV/EBITDA multiples
        "base_log_assets": 6.6,  # Mid-cap USD equivalent scale
        "vol_addon": 0.11,
    },
}


def _build_company_universe(n_companies: int, rng: np.random.Generator) -> List[Dict[str, str]]:
    """Create a multi-region universe covering US (NYSE/NASDAQ), Europe, and Turkey (BIST)."""
    universe: List[Dict[str, str]] = []
    real_tickers: List[Dict[str, str]] = []

    for region, sec_map in MARKET_TICKERS_BY_REGION_AND_SECTOR.items():
        curr = str(REGION_PARAMETERS[region]["currency"])
        for sector, tickers in sec_map.items():
            for t in tickers:
                real_tickers.append(
                    {
                        "ticker": t.upper(),
                        "company_name": f"{t.upper()} ({sector})",
                        "sector": sector,
                        "region": region,
                        "currency": curr,
                        "data_source": "Historical_Benchmark",
                    }
                )

    for item in real_tickers[: min(n_companies, len(real_tickers))]:
        universe.append(item)

    prefixes_by_region = {
        "US (NYSE/NASDAQ)": ("USX", "NSD", "NYT", "AMR"),
        "Europe (Euronext/XETRA/LSE)": ("EUX", "DAX", "PAR", "LON"),
        "Turkey (Borsa İstanbul - BIST)": ("BST", "IST", "TRK", "ANL"),
    }
    suffix_by_region = {
        "US (NYSE/NASDAQ)": "",
        "Europe (Euronext/XETRA/LSE)": ".DE",
        "Turkey (Borsa İstanbul - BIST)": ".IS",
    }

    idx = 0
    while len(universe) < n_companies:
        region = REGIONS[idx % len(REGIONS)]
        sector = SECTORS[idx % len(SECTORS)]
        prefixes = prefixes_by_region[region]
        prefix = prefixes[idx % len(prefixes)]
        suffix = suffix_by_region[region]
        ticker = f"{prefix}{100 + idx}{suffix}"
        universe.append(
            {
                "ticker": ticker,
                "company_name": f"{prefix} {sector.split()[0]} #{100 + idx} ({region.split()[0]})",
                "sector": sector,
                "region": region,
                "currency": str(REGION_PARAMETERS[region]["currency"]),
                "data_source": "Historical_Benchmark",
            }
        )
        idx += 1
    return universe


def generate_historical_ma_panel(
    n_companies: int = 420,
    years: Optional[List[int]] = None,
    target_rate: float = 0.055,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a multi-year, multi-region company-year financial panel dataset (standardized in USD Millions)
    with realistic M&A target labels across US, Europe, and Turkey (BIST).
    """
    if years is None:
        years = [2019, 2020, 2021, 2022, 2023, 2024]

    rng = np.random.default_rng(random_state)
    universe = _build_company_universe(n_companies, rng)

    records = []
    for comp in universe:
        sector = comp["sector"]
        region = comp["region"]
        prof = SECTOR_PROFILES[sector]
        reg_p = REGION_PARAMETERS[region]

        mult_scale = float(reg_p["multiple_scale"])
        vol_addon = float(reg_p["vol_addon"])
        base_log_assets = rng.normal(loc=float(reg_p["base_log_assets"]), scale=1.25)
        firm_quality = rng.normal(0.0, 1.0)
        is_structural_target = rng.random() < 0.22

        for yr_idx, year in enumerate(years):
            macro_shock = -0.04 if year == 2020 else (0.03 if year == 2021 else 0.0)

            log_assets = base_log_assets + 0.05 * yr_idx + rng.normal(0, 0.08)
            total_assets = float(np.exp(log_assets))  # in USD Millions

            if is_structural_target and rng.random() < 0.65:
                mismatch_regime = rng.choice(["high_growth_low_cash", "low_growth_high_cash"])
            else:
                mismatch_regime = "normal"

            if mismatch_regime == "high_growth_low_cash":
                rev_growth_1y = float(np.clip(prof["growth"] + rng.normal(0.14, 0.06), -0.15, 0.65))
                rev_growth_3y = float(np.clip(prof["growth"] + rng.normal(0.12, 0.04), -0.10, 0.45))
                cash_ratio = float(np.clip(rng.normal(0.035, 0.015), 0.01, 0.12))
                debt_ratio = float(np.clip(rng.normal(0.22, 0.07), 0.05, 0.45))
            elif mismatch_regime == "low_growth_high_cash":
                rev_growth_1y = float(np.clip(prof["growth"] - rng.normal(0.07, 0.04), -0.25, 0.12))
                rev_growth_3y = float(np.clip(prof["growth"] - rng.normal(0.05, 0.03), -0.15, 0.10))
                cash_ratio = float(np.clip(rng.normal(0.24, 0.05), 0.12, 0.42))
                debt_ratio = float(np.clip(rng.normal(0.12, 0.05), 0.01, 0.28))
            else:
                rev_growth_1y = float(np.clip(prof["growth"] + macro_shock + rng.normal(0.0, 0.08), -0.30, 0.50))
                rev_growth_3y = float(np.clip(prof["growth"] + rng.normal(0.0, 0.05), -0.20, 0.35))
                cash_ratio = float(np.clip(rng.normal(0.11, 0.05), 0.015, 0.35))
                debt_ratio = float(np.clip(rng.normal(0.30, 0.12), 0.02, 0.72))

            total_debt = total_assets * debt_ratio
            cash_and_equivalents = total_assets * cash_ratio
            total_equity = max(
                total_assets * float(np.clip(1.0 - debt_ratio - rng.uniform(0.15, 0.28), 0.12, 0.82)),
                10.0,
            )

            current_assets = max(
                cash_and_equivalents + total_assets * rng.uniform(0.12, 0.28),
                cash_and_equivalents * 1.1,
            )
            current_liabilities = max(total_assets * rng.uniform(0.10, 0.25), 4.0)

            asset_turnover = float(np.clip(rng.normal(0.78, 0.25), 0.20, 2.10))
            revenue = max(total_assets * asset_turnover, 15.0)
            revenue_t_minus_1 = revenue / max(1.0 + rev_growth_1y, 0.35)
            revenue_t_minus_3 = revenue / max((1.0 + rev_growth_3y) ** 3, 0.35)

            ebitda_margin = float(
                np.clip(prof["margin"] + 0.02 * firm_quality + macro_shock * 0.5 + rng.normal(0, 0.045), 0.04, 0.48)
            )
            ebitda = revenue * ebitda_margin
            ebit = ebitda * rng.uniform(0.72, 0.88)
            interest_expense = max(total_debt * rng.uniform(0.038, 0.068), 0.5)
            net_income = (ebit - interest_expense) * 0.78

            capex = revenue * float(np.clip(rng.normal(0.055, 0.022), 0.012, 0.16))
            fcf_conversion = rng.uniform(0.68, 1.08) if is_structural_target else rng.uniform(0.45, 0.92)
            operating_cash_flow = ebitda * fcf_conversion

            q_slope = rng.normal(-0.008 if is_structural_target else 0.002, 0.012)
            q1_m = float(np.clip(ebitda_margin - 1.5 * q_slope + rng.normal(0, 0.006), 0.02, 0.55))
            q2_m = float(np.clip(ebitda_margin - 0.5 * q_slope + rng.normal(0, 0.006), 0.02, 0.55))
            q3_m = float(np.clip(ebitda_margin + 0.5 * q_slope + rng.normal(0, 0.006), 0.02, 0.55))
            q4_m = float(np.clip(ebitda_margin + 1.5 * q_slope + rng.normal(0, 0.006), 0.02, 0.55))

            # Region-adjusted EV/EBITDA multiple (reflecting Country Risk Premium differences)
            discount_factor = rng.uniform(0.52, 0.82) if is_structural_target else rng.uniform(0.78, 1.38)
            regional_base_ev_ebitda = prof["ev_ebitda"] * mult_scale
            target_ev_ebitda = max(regional_base_ev_ebitda * discount_factor + rng.normal(0, 1.0 * mult_scale), 2.2)
            enterprise_value = max(ebitda * target_ev_ebitda, total_assets * 0.28)
            market_cap = max(enterprise_value - total_debt + cash_and_equivalents, total_equity * 0.40, 15.0)

            stock_return_12m = float(
                np.clip(
                    rng.normal(-0.14 if is_structural_target else 0.09, 0.20 + vol_addon * 0.5) + macro_shock,
                    -0.65,
                    1.25,
                )
            )
            stock_volatility_12m = float(
                np.clip(rng.normal((0.32 if is_structural_target else 0.26) + vol_addon, 0.08), 0.12, 0.85)
            )

            records.append(
                {
                    "ticker": comp["ticker"],
                    "company_name": comp["company_name"],
                    "region": region,
                    "currency": comp["currency"],
                    "data_source": comp["data_source"],
                    "sector": sector,
                    "fiscal_year": int(year),
                    "total_assets": round(total_assets, 2),
                    "total_equity": round(total_equity, 2),
                    "total_debt": round(total_debt, 2),
                    "cash_and_equivalents": round(cash_and_equivalents, 2),
                    "current_assets": round(current_assets, 2),
                    "current_liabilities": round(current_liabilities, 2),
                    "revenue": round(revenue, 2),
                    "revenue_t_minus_1": round(revenue_t_minus_1, 2),
                    "revenue_t_minus_3": round(revenue_t_minus_3, 2),
                    "ebitda": round(ebitda, 2),
                    "ebit": round(ebit, 2),
                    "net_income": round(net_income, 2),
                    "interest_expense": round(interest_expense, 2),
                    "operating_cash_flow": round(operating_cash_flow, 2),
                    "capex": round(capex, 2),
                    "market_cap": round(market_cap, 2),
                    "stock_return_12m": round(stock_return_12m, 4),
                    "stock_volatility_12m": round(stock_volatility_12m, 4),
                    "q1_ebitda_margin": round(q1_m, 4),
                    "q2_ebitda_margin": round(q2_m, 4),
                    "q3_ebitda_margin": round(q3_m, 4),
                    "q4_ebitda_margin": round(q4_m, 4),
                    "_structural_flag": int(is_structural_target),
                }
            )

    df = pd.DataFrame(records)

    # Calculate structural M&A propensity using REGION-SPECIFIC peer medians (preventing Country Risk Premium bias)
    ev = df["market_cap"] + df["total_debt"] - df["cash_and_equivalents"]
    ev_to_ebitda = ev / np.maximum(df["ebitda"], 1.0)
    sector_med_ev = df.groupby(["region", "sector", "fiscal_year"])["market_cap"].transform(
        lambda s: np.median(ev_to_ebitda.loc[s.index])
    )
    underval_score = (sector_med_ev - ev_to_ebitda) / np.maximum(sector_med_ev, 1.0)

    rev_g = (df["revenue"] - df["revenue_t_minus_1"]) / np.maximum(df["revenue_t_minus_1"], 1.0)
    cash_rat = df["cash_and_equivalents"] / np.maximum(df["total_assets"], 1.0)
    fcf_rat = (df["operating_cash_flow"] - df["capex"]) / np.maximum(df["total_assets"], 1.0)
    net_debt_ebitda = (df["total_debt"] - df["cash_and_equivalents"]) / np.maximum(df["ebitda"], 1.0)
    log_assets = np.log(np.maximum(df["total_assets"], 1.0))

    def _z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / max(s.std(), 1e-6)

    mismatch_signal = np.abs(_z(rev_g) - _z(cash_rat))
    midcap_sweet_spot = -0.35 * np.abs(log_assets - 7.3)

    lbo_sweet_spot = (
        (underval_score > 0.12) & (fcf_rat > fcf_rat.median()) & (net_debt_ebitda < 2.2)
    ).astype(float)
    palepu_corner = (
        ((rev_g > 0.16) & (cash_rat < 0.06)) | ((rev_g < 0.03) & (cash_rat > 0.18))
    ).astype(float)
    distress_debt_cliff = (net_debt_ebitda > 4.0).astype(float)

    latent_logit = (
        0.75 * _z(underval_score)
        + 0.45 * _z(mismatch_signal)
        + 0.45 * _z(fcf_rat)
        - 0.35 * _z(net_debt_ebitda)
        - 0.35 * _z(df["stock_return_12m"])
        + 0.35 * midcap_sweet_spot
        + 1.35 * lbo_sweet_spot
        + 1.15 * palepu_corner
        - 1.40 * distress_debt_cliff
        + 0.55 * df["_structural_flag"]
        + rng.normal(0.0, 0.65, size=len(df))
    )

    low_b, high_b = -10.0, 5.0
    for _ in range(40):
        mid_b = 0.5 * (low_b + high_b)
        probs = 1.0 / (1.0 + np.exp(-(latent_logit + mid_b)))
        if probs.mean() < target_rate:
            low_b = mid_b
        else:
            high_b = mid_b

    true_probs = 1.0 / (1.0 + np.exp(-(latent_logit + 0.5 * (low_b + high_b))))
    df["true_ma_propensity"] = np.round(true_probs, 4)
    df["is_acquired"] = (rng.random(len(df)) < true_probs).astype(int)
    df = df.drop(columns=["_structural_flag"])

    return df
