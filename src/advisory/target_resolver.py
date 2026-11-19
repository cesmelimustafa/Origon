"""Target profile resolver supporting both public universe tickers and custom private company parameters."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.api.schemas import REGION_DISPLAY_MAP_EN, REGION_DISPLAY_MAP_TR, SECTOR_DISPLAY_MAP_EN, SECTOR_DISPLAY_MAP_TR, sanitize_display_text
from src.data.benchmark_generator import REGION_PARAMETERS, SECTOR_PROFILES

SECTOR_DESCRIPTIONS: Dict[str, str] = {
    "Technology": "enterprise software cloud infrastructure cybersecurity semiconductors digital transformation SaaS automation artificial intelligence data analytics platform",
    "Healthcare": "biotechnology pharmaceuticals clinical diagnostics medical devices hospital networks healthcare services specialty therapeutics life sciences",
    "Industrials": "industrial automation aerospace defense machinery logistics supply chain engineering electrical equipment manufacturing infrastructure components",
    "Consumer Discretionary": "e-commerce specialty retail consumer brands automotive components apparel leisure hospitality omnichannel distribution digital marketplace",
    "Consumer Staples": "packaged food beverages household products personal care grocery distribution agricultural supply chain consumer essentials",
    "Energy": "integrated oil gas renewable power generation midstream pipelines LNG energy transition utilities grid storage",
    "Financials": "commercial banking wealth management asset management fintech payments insurance capital markets specialty finance",
    "Materials": "specialty chemicals advanced polymers industrial gases metals mining packaging materials construction aggregates sustainable materials",
}

REVERSE_REGION_MAP: Dict[str, str] = {
    "ABD Pazarı": "US (NYSE/NASDAQ)",
    "United States Market": "US (NYSE/NASDAQ)",
    "US": "US (NYSE/NASDAQ)",
    "Avrupa Pazarı": "Europe (Euronext/XETRA/LSE)",
    "European Market": "Europe (Euronext/XETRA/LSE)",
    "Europe": "Europe (Euronext/XETRA/LSE)",
    "Türkiye Borsa İstanbul": "Turkey (Borsa İstanbul - BIST)",
    "Turkey Borsa Istanbul": "Turkey (Borsa İstanbul - BIST)",
    "Turkey": "Turkey (Borsa İstanbul - BIST)",
}

REVERSE_SECTOR_MAP: Dict[str, str] = {
    **{v: k for k, v in SECTOR_DISPLAY_MAP_TR.items()},
    **{v: k for k, v in SECTOR_DISPLAY_MAP_EN.items()},
}


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    return fval if np.isfinite(fval) else None


def resolve_target_profile(
    ticker: Optional[str] = None,
    ranked_df: Optional[pd.DataFrame] = None,
    custom_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve either an existing public company from `ranked_df` or a user-entered custom/private company
    into a unified financial profile dictionary for Modules 2, 3, and 4.
    """
    if ticker and ranked_df is not None and not ranked_df.empty:
        t_clean = str(ticker).upper().strip()
        matches = ranked_df[ranked_df["ticker"].str.upper() == t_clean]
        if not matches.empty:
            row = matches.iloc[0]
            sec_code = str(row.get("sector", "Technology"))
            reg_code = str(row.get("region", "US (NYSE/NASDAQ)"))
            vol_12m = float(row.get("stock_volatility_12m", 0.28))
            ret_12m = float(row.get("stock_return_12m", -0.05))
            obs_dist = _finite_or_none(row.get("dist_from_52w_high"))
            if obs_dist is not None:
                # Observed drawdown from the 52-week high, price / high - 1.
                dist_52w = float(np.clip(obs_dist, -0.80, 0.0))
            else:
                dist_52w = float(np.clip(min(ret_12m - 0.12, -0.04), -0.65, -0.02))

            def _mult(col: str, default: float) -> float:
                """Unwinsorized display multiple, falling back to the imputed model value when not meaningful."""
                disp = _finite_or_none(row.get(f"disp_{col}"))
                if disp is not None:
                    return disp
                model_val = _finite_or_none(row.get(col))
                return model_val if model_val is not None else default

            return {
                "is_custom": False,
                "ticker": str(row["ticker"]),
                "company_name": sanitize_display_text(str(row.get("company_name", row["ticker"]))),
                "region_code": reg_code,
                "sector_code": sec_code,
                "currency": str(row.get("currency", "USD")),
                "revenue": float(row.get("revenue", 500.0)),
                "ebitda": float(row.get("ebitda", 90.0)),
                "ebitda_margin": float(row.get("ebitda_margin", 0.18)),
                "total_assets": float(row.get("total_assets", 1000.0)),
                "total_equity": float(row.get("total_equity", 450.0)),
                "total_debt": float(row.get("total_debt", 200.0)),
                "cash_and_equivalents": float(row.get("cash_and_equivalents", 90.0)),
                "market_cap": float(row.get("market_cap", 900.0)),
                "enterprise_value": float(row.get("enterprise_value", 1010.0)),
                "ev_to_ebitda": _mult("ev_to_ebitda", 11.2),
                "price_to_book": _mult("price_to_book", 2.4),
                "price_to_sales": _mult("price_to_sales", 1.8),
                "sector_ev_ebitda_discount": _mult("sector_ev_ebitda_discount", 0.15),
                "net_debt_to_ebitda": _mult("net_debt_to_ebitda", 1.22),
                "fcf_to_assets": float(row.get("fcf_to_assets", 0.08)),
                "cash_to_assets": float(row.get("cash_to_assets", 0.09)),
                "rev_growth_1y": float(row.get("rev_growth_1y", 0.10)),
                "rev_growth_3y_cagr": float(row.get("rev_growth_3y_cagr", 0.09)),
                "roa": float(row.get("roa", 0.07)),
                "roe": float(row.get("roe", 0.14)),
                "interest_coverage": float(row.get("interest_coverage", 6.5)),
                "stock_return_12m": ret_12m,
                "stock_volatility_12m": vol_12m,
                "stock_volatility_90d": round(float(np.clip(vol_12m * 1.05, 0.12, 0.85)), 4),
                "dist_from_52w_high": round(dist_52w, 4),
                "predicted_ma_pct": float(row.get("predicted_ma_pct", 35.0)),
                "business_description": f"{sec_code} {SECTOR_DESCRIPTIONS.get(sec_code, '')}",
            }

    payload = custom_payload or {}
    raw_reg = str(payload.get("region", "US (NYSE/NASDAQ)"))
    reg_code = REVERSE_REGION_MAP.get(raw_reg, raw_reg if raw_reg in REGION_PARAMETERS else "US (NYSE/NASDAQ)")

    raw_sec = str(payload.get("sector", "Technology"))
    sec_code = REVERSE_SECTOR_MAP.get(raw_sec, raw_sec if raw_sec in SECTOR_PROFILES else "Technology")

    prof = SECTOR_PROFILES[sec_code]
    mult_scale = float(REGION_PARAMETERS[reg_code]["multiple_scale"])
    regional_sector_ev_ebitda = prof["ev_ebitda"] * mult_scale

    revenue = max(float(payload.get("revenue", 500.0)), 10.0)
    ebitda = max(float(payload.get("ebitda", revenue * prof["margin"])), 2.0)
    ebitda_margin = float(np.clip(ebitda / revenue, 0.02, 0.75))
    total_debt = max(float(payload.get("total_debt", ebitda * 1.8)), 0.0)
    cash = max(float(payload.get("cash_and_equivalents", revenue * 0.10)), 1.0)

    # For private companies where market_cap is 0 or omitted, derive baseline equity from regional sector multiple
    input_mcap = float(payload.get("market_cap", 0.0))
    if input_mcap <= 0:
        implied_base_ev = ebitda * regional_sector_ev_ebitda * 0.88
        market_cap = max(implied_base_ev - total_debt + cash, ebitda * 3.0, 25.0)
    else:
        market_cap = max(input_mcap, 10.0)

    enterprise_value = max(market_cap + total_debt - cash, ebitda * 2.5)
    ev_to_ebitda = float(np.clip(enterprise_value / ebitda, 1.5, 85.0))
    total_assets = max(float(payload.get("total_assets", enterprise_value * 0.92)), revenue * 0.6, 20.0)
    total_equity = max(float(payload.get("total_equity", total_assets - total_debt)), market_cap * 0.45, 15.0)
    price_to_book = float(np.clip(market_cap / total_equity, 0.2, 45.0))
    price_to_sales = float(np.clip(market_cap / revenue, 0.1, 35.0))

    sector_discount = float(
        np.clip((regional_sector_ev_ebitda - ev_to_ebitda) / max(regional_sector_ev_ebitda, 1.0), -1.5, 0.85)
    )
    net_debt_to_ebitda = float(np.clip((total_debt - cash) / ebitda, -6.0, 20.0))
    fcf = float(payload.get("free_cash_flow", ebitda * 0.58))
    fcf_to_assets = float(np.clip(fcf / total_assets, -0.35, 0.45))
    cash_to_assets = float(np.clip(cash / total_assets, 0.01, 0.75))
    rev_growth_1y = float(np.clip(float(payload.get("rev_growth_1y", prof["growth"])), -0.50, 1.50))
    vol_90d = float(np.clip(float(payload.get("stock_volatility_90d", 0.29)), 0.10, 0.85))
    dist_52w = float(np.clip(float(payload.get("dist_from_52w_high", -0.18)), -0.75, 0.0))
    custom_desc = str(payload.get("business_description", "")).strip()

    comp_name = sanitize_display_text(str(payload.get("company_name", "Özel Hedef Şirket")))
    ticker_sym = sanitize_display_text(str(payload.get("ticker", "CUSTOM"))).upper()[:10] or "CUSTOM"

    return {
        "is_custom": True,
        "ticker": ticker_sym,
        "company_name": comp_name,
        "region_code": reg_code,
        "sector_code": sec_code,
        "currency": "USD",
        "revenue": round(revenue, 2),
        "ebitda": round(ebitda, 2),
        "ebitda_margin": round(ebitda_margin, 4),
        "total_assets": round(total_assets, 2),
        "total_equity": round(total_equity, 2),
        "total_debt": round(total_debt, 2),
        "cash_and_equivalents": round(cash, 2),
        "market_cap": round(market_cap, 2),
        "enterprise_value": round(enterprise_value, 2),
        "ev_to_ebitda": round(ev_to_ebitda, 2),
        "price_to_book": round(price_to_book, 2),
        "price_to_sales": round(price_to_sales, 2),
        "sector_ev_ebitda_discount": round(sector_discount, 4),
        "net_debt_to_ebitda": round(net_debt_to_ebitda, 2),
        "fcf_to_assets": round(fcf_to_assets, 4),
        "cash_to_assets": round(cash_to_assets, 4),
        "rev_growth_1y": round(rev_growth_1y, 4),
        "rev_growth_3y_cagr": round(rev_growth_1y * 0.85, 4),
        "roa": round((ebitda * 0.52) / total_assets, 4),
        "roe": round((ebitda * 0.52) / total_equity, 4),
        "interest_coverage": round(ebitda * 0.82 / max(total_debt * 0.055, 1.0), 2),
        "stock_return_12m": round(dist_52w * 0.85, 4),
        "stock_volatility_12m": round(vol_90d, 4),
        "stock_volatility_90d": round(vol_90d, 4),
        "dist_from_52w_high": round(dist_52w, 4),
        "predicted_ma_pct": 42.5,
        "business_description": f"{sec_code} {custom_desc} {SECTOR_DESCRIPTIONS.get(sec_code, '')}",
    }
