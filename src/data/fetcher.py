"""Multi-region (US, Europe, Turkey BIST) live yfinance data fetcher with FX normalization to USD."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DEFAULT_FX_TO_USD, MARKET_TICKERS_BY_REGION_AND_SECTOR
from src.data.benchmark_generator import SECTOR_PROFILES, generate_historical_ma_panel

logger = logging.getLogger(__name__)

MONETARY_COLUMNS: List[str] = [
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
]


def convert_row_currency_to_usd(
    row: Dict[str, Any],
    fx_to_usd: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Convert all monetary balance-sheet, income-statement, and market-cap fields of a company
    from its reporting currency (e.g. TRY, EUR, GBP, CHF) into USD Millions so cross-market
    scale features (`log_total_assets`, `market_cap`) are apples-to-apples.
    """
    fx_map = fx_to_usd or DEFAULT_FX_TO_USD
    out = dict(row)
    curr = str(out.get("currency") or "USD")
    rate = float(fx_map.get(curr, fx_map.get(curr.upper(), 1.0)))

    for col in MONETARY_COLUMNS:
        if col in out and out[col] is not None:
            out[col] = round(float(out[col]) * rate, 2)

    out["reporting_currency"] = curr
    out["fx_rate_to_usd"] = round(rate, 6)
    return out


class YFinanceDataFetcher:
    """
    Fetches live corporate financial statements and market multiples from Yahoo Finance (`yfinance`)
    across US (NYSE/NASDAQ), Europe (Euronext/XETRA/LSE), and Turkey (Borsa İstanbul - BIST),
    converting all balance sheets to USD Millions via live FX rates and merging with historical
    training panels.
    """

    def __init__(self, cache_path: Optional[Path] = None, cache_ttl_seconds: int = 3600):
        self.cache_path = cache_path or (ARTIFACTS_DIR / "market_panel_cache.parquet")
        self.cache_ttl_seconds = cache_ttl_seconds

    def fetch_live_fx_rates(self) -> Dict[str, float]:
        """Fetch live FX conversion rates to USD (1 unit of currency -> USD)."""
        fx_rates = dict(DEFAULT_FX_TO_USD)
        try:
            import yfinance as yf

            pairs = {
                "EUR": ("EURUSD=X", False),
                "GBP": ("GBPUSD=X", False),
                "CHF": ("CHFUSD=X", False),
                "TRY": ("TRY=X", True),   # USD/TRY -> invert
                "SEK": ("SEK=X", True),   # USD/SEK -> invert
                "NOK": ("NOK=X", True),   # USD/NOK -> invert
            }
            for curr, (pair_sym, invert) in pairs.items():
                try:
                    fi = yf.Ticker(pair_sym).fast_info
                    last_px = float(fi.get("lastPrice") or fi.get("regularMarketPrice") or 0.0)
                    if last_px > 0:
                        fx_rates[curr] = (1.0 / last_px) if invert else last_px
                except Exception:
                    continue
            fx_rates["GBp"] = fx_rates["GBP"] * 0.01
        except Exception as exc:
            logger.debug("Using fallback FX rates due to: %s", exc)
        return fx_rates

    def _fetch_single_ticker(
        self,
        region: str,
        sector: str,
        ticker_sym: str,
        fiscal_year: int,
        fx_rates: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        try:
            import yfinance as yf

            tk = yf.Ticker(ticker_sym)
            info = tk.info or {}
            raw_mcap = float(info.get("marketCap") or 0.0) / 1e6
            if raw_mcap <= 0:
                return None

            fin_curr = str(info.get("financialCurrency") or info.get("currency") or "USD")
            mkt_curr = str(info.get("currency") or fin_curr)

            # Convert market_cap first if market currency differs from financial statement currency (e.g. LSE GBp vs GBP/USD)
            mkt_to_usd = float(fx_rates.get(mkt_curr, fx_rates.get(mkt_curr.upper(), 1.0)))
            fin_to_usd = float(fx_rates.get(fin_curr, fx_rates.get(fin_curr.upper(), 1.0)))
            mcap_in_fin_curr = raw_mcap * (mkt_to_usd / max(fin_to_usd, 1e-6))

            def _info_m(key: str) -> Optional[float]:
                """Return an info field in millions, or None when Yahoo Finance does not report it."""
                val = info.get(key)
                if val is None:
                    return None
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(fval):
                    return None
                return fval / 1e6

            def _info_ratio(key: str) -> Optional[float]:
                val = info.get(key)
                try:
                    fval = float(val) if val is not None else None
                except (TypeError, ValueError):
                    return None
                return fval if fval is not None and np.isfinite(fval) else None

            # All statement fields below are already expressed in millions of the financial currency.
            total_assets = _info_m("totalAssets") or (mcap_in_fin_curr * 1.15)
            total_debt = _info_m("totalDebt") or 0.0
            cash = _info_m("totalCash") or 0.0
            revenue = _info_m("totalRevenue") or (mcap_in_fin_curr * 0.65)
            revenue = max(revenue, 1.0)

            # EBITDA fallback chain: reported EBITDA, then EBITDA margin, then operating margin plus a
            # depreciation proxy, then the sector profile margin. Reported negative EBITDA is preserved
            # so that EV/EBITDA can be flagged as not meaningful downstream instead of being fabricated.
            reported_ebitda = _info_m("ebitda")
            ebitda_margin_info = _info_ratio("ebitdaMargins")
            op_margin_info = _info_ratio("operatingMargins")
            if reported_ebitda is not None and reported_ebitda != 0.0:
                ebitda = reported_ebitda
            elif ebitda_margin_info is not None and ebitda_margin_info != 0.0:
                ebitda = revenue * ebitda_margin_info
            elif op_margin_info is not None and op_margin_info != 0.0:
                ebitda = revenue * (op_margin_info + 0.035)
            else:
                ebitda = revenue * float(SECTOR_PROFILES.get(sector, {}).get("margin", 0.15))

            net_income_m = _info_m("netIncomeToCommon")
            net_income = net_income_m if net_income_m is not None else ebitda * 0.55
            fcf_m = _info_m("freeCashflow")
            fcf = fcf_m if fcf_m is not None else ebitda * 0.48
            rev_growth = float(np.clip(float(info.get("revenueGrowth") or 0.07), -0.60, 1.50))

            # For high-inflation reporting currency (TRY), deflate nominal TRY revenue growth to real USD-equivalent growth
            if fin_curr.upper() == "TRY" and rev_growth > 0.30:
                rev_growth = float(np.clip((1.0 + rev_growth) / 1.38 - 1.0, -0.35, 0.55))

            book_val = float(info.get("bookValue") or 10.0)
            shares = float(info.get("sharesOutstanding") or 1e7)
            total_equity = max((book_val * shares) / 1e6, total_assets - total_debt, total_assets * 0.20, 10.0)

            ebitda_margin = float(np.clip(ebitda / max(revenue, 1.0), -0.50, 0.60))

            # Observed drawdown from the 52-week high, used as the Baker, Pan and Wurgler reference point.
            last_px = _info_ratio("currentPrice") or _info_ratio("regularMarketPrice")
            high_52w = _info_ratio("fiftyTwoWeekHigh")
            if last_px and high_52w and high_52w > 0:
                dist_52w_high = float(np.clip(last_px / high_52w - 1.0, -0.90, 0.0))
            else:
                dist_52w_high = np.nan
            raw_row = {
                "ticker": ticker_sym.upper(),
                "company_name": info.get("shortName") or info.get("longName") or f"{ticker_sym.upper()} ({sector})",
                "region": region,
                "currency": fin_curr,
                "data_source": "Live_Yahoo_Finance",
                "sector": sector,
                "fiscal_year": fiscal_year,
                "total_assets": total_assets,
                "total_equity": total_equity,
                "total_debt": total_debt,
                "cash_and_equivalents": cash,
                "current_assets": max(cash * 1.8, total_assets * 0.25),
                "current_liabilities": max(total_assets * 0.18, 5.0),
                "revenue": revenue,
                "revenue_t_minus_1": revenue / max(1.0 + rev_growth, 0.4),
                "revenue_t_minus_3": revenue / max((1.0 + rev_growth) ** 3, 0.4),
                "ebitda": ebitda,
                "ebit": ebitda - revenue * 0.035,
                "net_income": net_income,
                "interest_expense": max(total_debt * 0.05, 0.5),
                "operating_cash_flow": fcf + revenue * 0.045,
                "capex": revenue * 0.045,
                "market_cap": mcap_in_fin_curr,
                "stock_return_12m": round(float(np.clip(float(info.get("52WeekChange") or 0.04), -0.75, 1.80)), 4),
                "stock_volatility_12m": round(float(np.clip(float(info.get("beta") or 1.05) * 0.24, 0.12, 0.85)), 4),
                "q1_ebitda_margin": round(ebitda_margin, 4),
                "q2_ebitda_margin": round(ebitda_margin, 4),
                "q3_ebitda_margin": round(ebitda_margin, 4),
                "q4_ebitda_margin": round(ebitda_margin, 4),
                "dist_from_52w_high": dist_52w_high,
                "true_ma_propensity": np.nan,
                "is_acquired": 0,
            }
            return convert_row_currency_to_usd(raw_row, fx_to_usd=fx_rates)
        except Exception as exc:
            logger.debug("Skipping ticker %s due to fetch error: %s", ticker_sym, exc)
            return None

    def fetch_live_snapshots(
        self,
        tickers_by_region_and_sector: Optional[Dict[str, Dict[str, List[str]]]] = None,
        fiscal_year: int = 2024,
        max_tickers_per_sector: int = 3,
        max_workers: int = 8,
    ) -> pd.DataFrame:
        """
        Concurrently pull live fundamentals across US, Europe, and Turkey (BIST) via `yfinance`
        and normalize all monetary figures to USD Millions.
        """
        try:
            import yfinance as _  # noqa: F401
        except ImportError:
            logger.warning("yfinance not installed; skipping live fetch.")
            return pd.DataFrame()

        region_map = tickers_by_region_and_sector or MARKET_TICKERS_BY_REGION_AND_SECTOR
        fx_rates = self.fetch_live_fx_rates()

        tasks: List[Tuple[str, str, str]] = []
        for region, sec_map in region_map.items():
            for sector, tickers in sec_map.items():
                for ticker_sym in tickers[:max_tickers_per_sector]:
                    tasks.append((region, sector, ticker_sym))

        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self._fetch_single_ticker, reg, sec, sym, fiscal_year, fx_rates)
                for reg, sec, sym in tasks
            ]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    rows.append(res)

        return pd.DataFrame(rows)

    def load_or_build_panel(
        self,
        n_companies: int = 420,
        years: Optional[List[int]] = None,
        use_live_yfinance: bool = True,
        force_rebuild: bool = False,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Load cached panel (if within TTL and not forced) or construct the multi-region
        historical M&A training panel + live Yahoo Finance active company overlay.
        """
        if self.cache_path.exists() and not force_rebuild:
            age_sec = time.time() - self.cache_path.stat().st_mtime
            if age_sec <= self.cache_ttl_seconds:
                cached_df = pd.read_parquet(self.cache_path)
                if "region" in cached_df.columns:
                    return cached_df

        panel_df = generate_historical_ma_panel(
            n_companies=n_companies,
            years=years or [2019, 2020, 2021, 2022, 2023, 2024],
            target_rate=0.055,
            random_state=random_state,
        )

        if use_live_yfinance:
            latest_year = int(max(panel_df["fiscal_year"]))
            live_df = self.fetch_live_snapshots(fiscal_year=latest_year)
            if not live_df.empty:
                # Preserve historical acquisition outcome flags for backtest evaluation while updating live fundamentals
                hist_outcomes = (
                    panel_df[panel_df["fiscal_year"] == latest_year]
                    .set_index("ticker")["is_acquired"]
                    .to_dict()
                )
                live_df["is_acquired"] = [
                    int(hist_outcomes.get(t, 0)) for t in live_df["ticker"]
                ]
                existing_keys = set(zip(live_df["ticker"], live_df["fiscal_year"]))
                mask = [
                    (t, y) not in existing_keys
                    for t, y in zip(panel_df["ticker"], panel_df["fiscal_year"])
                ]
                panel_df = pd.concat([panel_df[mask], live_df], ignore_index=True)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel_df.to_parquet(self.cache_path, index=False)
        return panel_df
