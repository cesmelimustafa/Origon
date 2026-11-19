"""Financial feature engineering across the 5 M&A corporate finance hypothesis categories."""

from typing import List
import numpy as np
import pandas as pd

EPS = 1e-3


def _safe_pos(series: pd.Series, floor: float = EPS) -> pd.Series:
    return np.maximum(pd.to_numeric(series, errors="coerce").fillna(floor), floor)


def _peer_group_keys(df: pd.DataFrame) -> List[str]:
    """Group by (region, sector, fiscal_year) when region is available to prevent cross-market ERP bias."""
    if "region" in df.columns:
        return ["region", "sector", "fiscal_year"]
    return ["sector", "fiscal_year"]


def _robust_group_z(s: pd.Series) -> pd.Series:
    med = s.median()
    iqr = float(s.quantile(0.75) - s.quantile(0.25))
    scale = iqr if iqr > 1e-5 else max(float(s.std()) if len(s) > 1 else 0.0, 1e-3)
    return ((s - med) / scale).fillna(0.0)


def palepu_growth_resource_mismatch(
    df: pd.DataFrame,
    group_keys: List[str],
    growth_col: str = "rev_growth_1y",
    liquidity_col: str = "cash_to_assets",
    leverage_col: str = "debt_to_equity",
) -> pd.Series:
    """
    Continuous version of the Palepu 1986 growth-resource imbalance hypothesis.

    Palepu defines the imbalance dummy as one for firms with high growth, low liquidity and high
    leverage, or low growth, high liquidity and low leverage. Using peer robust Z-scores:
        R  = 0.5 * (Z_liquidity - Z_leverage)      resource richness
        GRM = |Z_growth - R|                         clipped to [0, 8]
    A high growth, resource poor firm has Z_growth >> R and a low growth, resource rich firm has
    Z_growth << R; both produce a large index, while balanced firms score near zero.
    """
    g_z = df.groupby(group_keys)[growth_col].transform(_robust_group_z).clip(-4.0, 4.0)
    l_z = df.groupby(group_keys)[liquidity_col].transform(_robust_group_z).clip(-4.0, 4.0)
    lev_z = df.groupby(group_keys)[leverage_col].transform(_robust_group_z).clip(-4.0, 4.0)
    resource_richness = 0.5 * (l_z - lev_z)
    return np.abs(g_z - resource_richness).clip(lower=0.0, upper=8.0)


def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 22 corporate finance ratios and interaction terms across the 5 M&A hypotheses.
    Peer-relative valuation discounts, growth-resource mismatch, and HHI concentration are computed
    within each (region, sector, fiscal_year) cell so US, European, and Turkish (BIST) companies
    are benchmarked against their own regional market regime.
    """
    out = df.copy()
    group_keys = _peer_group_keys(out)

    assets = _safe_pos(out["total_assets"], 1.0)
    equity = _safe_pos(out["total_equity"], 1.0)
    revenue = _safe_pos(out["revenue"], 1.0)
    rev_t1 = _safe_pos(out["revenue_t_minus_1"], 1.0)
    rev_t3 = _safe_pos(out["revenue_t_minus_3"], 1.0)
    ebitda_raw = pd.to_numeric(out["ebitda"], errors="coerce")
    # EBITDA based multiples are only meaningful for positive EBITDA. Market practice and
    # Damodaran style valuation reports flag them as NM; here they become NaN and are imputed
    # with the training sector median by the leakage free preprocessor.
    ebitda_meaningful = ebitda_raw > max(EPS, 0.0)
    ebitda = ebitda_raw.where(ebitda_meaningful)
    curr_liab = _safe_pos(out["current_liabilities"], 1.0)
    interest_exp = _safe_pos(out["interest_expense"], 0.1)

    total_debt = pd.to_numeric(out["total_debt"], errors="coerce").fillna(0.0)
    cash = pd.to_numeric(out["cash_and_equivalents"], errors="coerce").fillna(0.0)
    market_cap = _safe_pos(out["market_cap"], 1.0)

    # 1. Undervaluation (Region-Sector peer relative)
    enterprise_value = np.maximum(market_cap + total_debt - cash, 1.0)
    out["enterprise_value"] = enterprise_value
    out["ebitda_meaningful"] = ebitda_meaningful.astype(int)
    out["ev_to_ebitda"] = (enterprise_value / ebitda).clip(lower=0.5, upper=120.0)
    out["price_to_book"] = (market_cap / equity).clip(lower=0.1, upper=80.0)
    out["price_to_sales"] = (market_cap / revenue).clip(lower=0.05, upper=60.0)

    # Peer medians use only meaningful observations; groupby median skips NaN.
    sec_ev_med = out.groupby(group_keys)["ev_to_ebitda"].transform("median")
    sec_pb_med = out.groupby(group_keys)["price_to_book"].transform("median")

    out["sector_ev_ebitda_discount"] = (
        (sec_ev_med - out["ev_to_ebitda"]) / np.maximum(sec_ev_med, EPS)
    ).clip(lower=-3.0, upper=0.95)
    out["sector_pb_discount"] = (
        (sec_pb_med - out["price_to_book"]) / np.maximum(sec_pb_med, EPS)
    ).clip(lower=-3.0, upper=0.95)

    # 2. Growth-Resource Mismatch
    out["rev_growth_1y"] = ((revenue - rev_t1) / rev_t1).clip(lower=-0.80, upper=2.50)
    out["rev_growth_3y_cagr"] = ((revenue / rev_t3) ** (1.0 / 3.0) - 1.0).clip(lower=-0.60, upper=1.50)

    fcf = pd.to_numeric(out["operating_cash_flow"], errors="coerce").fillna(0.0) - pd.to_numeric(
        out["capex"], errors="coerce"
    ).fillna(0.0)
    out["free_cash_flow"] = fcf
    out["fcf_to_assets"] = (fcf / assets).clip(lower=-0.60, upper=0.60)
    out["cash_to_assets"] = (cash / assets).clip(lower=0.0, upper=0.90)
    out["capex_to_revenue"] = (
        pd.to_numeric(out["capex"], errors="coerce").fillna(0.0) / revenue
    ).clip(lower=0.0, upper=0.80)

    # 3. Leverage & Liquidity
    out["net_debt_to_ebitda"] = ((total_debt - cash) / ebitda).clip(lower=-10.0, upper=25.0)
    out["debt_to_equity"] = (total_debt / equity).clip(lower=0.0, upper=25.0)
    out["current_ratio"] = (
        pd.to_numeric(out["current_assets"], errors="coerce").fillna(0.0) / curr_liab
    ).clip(lower=0.1, upper=15.0)
    out["interest_coverage"] = (
        pd.to_numeric(out["ebit"], errors="coerce").fillna(0.0) / interest_exp
    ).clip(lower=-20.0, upper=100.0)

    # Palepu 1986 growth-resource imbalance: a firm is mismatched when it is high growth but
    # resource poor, meaning low liquidity and high leverage, or low growth but resource rich,
    # meaning high liquidity and low leverage. Resource richness R = 0.5 * (Z_liquidity - Z_leverage)
    # and the continuous imbalance index is |Z_growth - R|, which is zero for balanced firms.
    out["growth_resource_mismatch"] = palepu_growth_resource_mismatch(
        out, group_keys, growth_col="rev_growth_1y", liquidity_col="cash_to_assets", leverage_col="debt_to_equity"
    )

    # 4. Profitability & Operational Efficiency
    ebitda_signed = ebitda_raw.fillna(0.0)
    out["ebitda_margin"] = (ebitda_signed / revenue).clip(lower=-0.50, upper=0.80)
    out["roa"] = (
        pd.to_numeric(out["net_income"], errors="coerce").fillna(0.0) / assets
    ).clip(lower=-0.60, upper=0.60)
    out["roe"] = (
        pd.to_numeric(out["net_income"], errors="coerce").fillna(0.0) / equity
    ).clip(lower=-1.00, upper=1.20)
    out["asset_turnover"] = (revenue / assets).clip(lower=0.05, upper=5.0)

    q1 = pd.to_numeric(out["q1_ebitda_margin"], errors="coerce").fillna(out["ebitda_margin"])
    q2 = pd.to_numeric(out["q2_ebitda_margin"], errors="coerce").fillna(out["ebitda_margin"])
    q3 = pd.to_numeric(out["q3_ebitda_margin"], errors="coerce").fillna(out["ebitda_margin"])
    q4 = pd.to_numeric(out["q4_ebitda_margin"], errors="coerce").fillna(out["ebitda_margin"])
    out["margin_momentum_4q"] = ((-1.5 * q1 - 0.5 * q2 + 0.5 * q3 + 1.5 * q4) / 5.0).clip(
        lower=-0.25, upper=0.25
    )

    # 5. Scale & Market Dynamics
    out["log_total_assets"] = np.log(assets)
    out["stock_return_12m"] = pd.to_numeric(out["stock_return_12m"], errors="coerce").fillna(0.0).clip(-0.90, 2.50)
    out["stock_volatility_12m"] = (
        pd.to_numeric(out["stock_volatility_12m"], errors="coerce").fillna(0.25).clip(0.05, 1.50)
    )

    sec_total_rev = out.groupby(group_keys)["revenue"].transform("sum")
    rev_share_sq = (revenue / np.maximum(sec_total_rev, 1.0)) ** 2
    out["sector_hhi"] = rev_share_sq.groupby([out[k] for k in group_keys]).transform("sum").clip(0.01, 1.0)

    return out
