"""Module 3: Acquisition Premium and Offer Price Recommendation Engine using Quantile Gradient Boosting."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.api.schemas import get_sector_map, is_en_lang, sanitize_display_text
from src.data.benchmark_generator import REGION_PARAMETERS, SECTOR_PROFILES

SECTOR_BASE_PREMIUMS: Dict[str, float] = {
    "Technology": 34.0,
    "Healthcare": 38.0,
    "Industrials": 27.5,
    "Consumer Discretionary": 29.5,
    "Consumer Staples": 26.5,
    "Energy": 23.5,
    "Financials": 25.0,
    "Materials": 26.0,
}

PRECEDENT_ACQUIRER_COUNTERPARTIES: Dict[str, List[Tuple_str := str]] = {
    "Technology": [
        ("Palo Alto Cloud Systems", "Thoma Bravo Software"),
        ("Vertex Enterprise AI", "Microsoft Strategic Corp"),
        ("Alpine Cyber Analytics", "Vista Equity Partners"),
        ("Nordic DevOps Platform", "SAP SE"),
        ("Anatolia ERP Solutions", "Francisco Partners"),
        ("Sentinel Data Networks", "Cisco Systems"),
    ],
    "Healthcare": [
        ("Helix Clinical Diagnostics", "Johnson and Johnson"),
        ("Bosphorus Hospital Group", "Turkven Private Equity"),
        ("Alpine Biopharma Labs", "Novartis AG"),
        ("Precision MedTech Devices", "EQT Capital Partners"),
        ("Apex Oncology Therapeutics", "Pfizer Inc"),
        ("Nordic Life Sciences", "Nordic Capital"),
    ],
    "Industrials": [
        ("Rhein Automation Robotics", "Siemens AG"),
        ("Aegean Precision Machinery", "Actera Group"),
        ("Midwest Aerospace Components", "Honeywell International"),
        ("Continental Grid Systems", "Eaton Corporation"),
        ("Baltic Logistics Engineering", "Triton Partners"),
        ("Atlas Fluid Controls", "KKR Buyout"),
    ],
    "Consumer Discretionary": [
        ("Milan Heritage Apparel", "LVMH Group"),
        ("Marmara Omnichannel Retail", "Esas Private Equity"),
        ("Urban Athletic Brands", "Leonard Green Partners"),
        ("Nordic Home Commerce", "Permira Buyout"),
        ("Pacific Mobility Parts", "Bain Capital"),
        ("Apex Digital Marketplace", "Apax Partners"),
    ],
    "Consumer Staples": [
        ("Anatolia Packaged Foods", "Coca-Cola Icecek"),
        ("Alpine Nutrition Ingredients", "Nestle SA"),
        ("Atlantic Beverage Bottling", "PepsiCo Inc"),
        ("Continental Grocery Supply", "CVC Capital Partners"),
        ("Nordic Organic Essentials", "PAI Partners"),
        ("Midwest Household Brands", "Procter and Gamble"),
    ],
    "Energy": [
        ("Permian Midstream Pipelines", "Exxon Mobil"),
        ("Aegean Renewable Power", "Tupras Energy"),
        ("North Sea LNG Terminals", "TotalEnergies SE"),
        ("Gulf Storage Infrastructure", "Brookfield Partners"),
        ("Iberian Solar Utilities", "Chevron Corporation"),
        ("Alpine Grid Storage", "Warburg Pincus"),
    ],
    "Financials": [
        ("Continental Wealth Advisory", "JPMorgan Strategic"),
        ("Nordic Payment Clearing", "Hg Capital"),
        ("Bosphorus Leasing and Factoring", "Sabanci Holding"),
        ("Atlantic Risk Analytics", "BlackRock Holdings"),
        ("Alpine Insurance Brokerage", "Hellman and Friedman"),
        ("Metro Digital Banking", "BNP Paribas"),
    ],
    "Materials": [
        ("Rhein Specialty Polymers", "BASF SE"),
        ("Anatolia Industrial Minerals", "OYAK Industrial"),
        ("Atlantic Industrial Gases", "Linde plc"),
        ("Nordic Sustainable Packaging", "Ardian Buyout"),
        ("Midwest Advanced Coatings", "Advent International"),
        ("Alpine Catalyst Materials", "Apollo Buyout"),
    ],
}

PREMIUM_FEATURE_COLS: List[str] = [
    "ev_to_ebitda",
    "price_to_book",
    "sector_ev_ebitda_discount",
    "stock_volatility_90d",
    "dist_from_52w_high",
    "interest_rate_env",
    "rev_growth_1y",
    "fcf_to_assets",
    "net_debt_to_ebitda",
]


def reference_point_premium_pct(dist_from_52w_high: float) -> float:
    """
    Exact offer premium percentage required to restore the target's 52-week high price
    (Baker, Pan & Wurgler, 2012, Journal of Financial Economics).

    Given drawdown d = (P_0 - P_52w) / P_52w <= 0, the required run-up to reach P_52w is:
        pi_52w* = (P_52w - P_0) / P_0 * 100 = (-d / (1 + d)) * 100
    """
    d = float(dist_from_52w_high)
    if not np.isfinite(d) or d >= 0.0:
        return 0.0
    d_clamped = max(d, -0.85)
    return float((-d_clamped / (1.0 + d_clamped)) * 100.0)


def board_acceptance_probability(
    premium_pct: float,
    inflection: float,
    spread: float,
    reference_premium: float = 0.0,
) -> float:
    """
    Monotone logistic board acceptance probability with the Baker, Pan & Wurgler (2012)
    discontinuous reference-point jump once the offer crosses the 52-week high price.
    """
    s = max(float(spread), 1.0)
    logit = (float(premium_pct) - float(inflection)) / s
    if reference_premium > 0.0 and float(premium_pct) >= float(reference_premium):
        logit += 0.58
    return float(np.clip(100.0 / (1.0 + np.exp(-logit)), 1.0, 99.2))


def generate_precedent_transactions_dataset(
    n_transactions: int = 640,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a calibrated historical M&A transactions dataset reflecting empirical corporate finance
    relationships between control premiums, valuation discounts, 52-week high anchors, volatility,
    growth rates, and interest rate environments.
    """
    rng = np.random.default_rng(random_state)
    sectors = list(SECTOR_PROFILES.keys())
    regions = list(REGION_PARAMETERS.keys())
    years = [2019, 2020, 2021, 2022, 2023, 2024]

    rows = []
    for i in range(n_transactions):
        sec = sectors[i % len(sectors)]
        reg = regions[i % len(regions)]
        yr = int(rng.choice(years))
        prof = SECTOR_PROFILES[sec]
        mult_scale = float(REGION_PARAMETERS[reg]["multiple_scale"])

        # Macro interest rate regime by year
        rate_map = {2019: 0.025, 2020: 0.010, 2021: 0.012, 2022: 0.038, 2023: 0.052, 2024: 0.048}
        interest_rate_env = rate_map[yr] + (0.04 if "Turkey" in reg else 0.0)

        sec_med_ev = prof["ev_ebitda"] * mult_scale
        ev_to_ebitda = float(np.clip(rng.normal(sec_med_ev * 0.86, 2.8 * mult_scale), 3.0, 38.0))
        price_to_book = float(np.clip(rng.normal(prof["pb"] * mult_scale * 0.9, 1.1), 0.6, 18.0))
        sector_discount = float(np.clip((sec_med_ev - ev_to_ebitda) / max(sec_med_ev, 1.0), -0.80, 0.70))

        vol_90d = float(np.clip(rng.normal(0.29, 0.09), 0.12, 0.75))
        dist_52w = float(np.clip(rng.normal(-0.19, 0.12), -0.65, -0.01))
        rev_growth_1y = float(np.clip(rng.normal(prof["growth"] + 0.03, 0.09), -0.20, 0.55))
        fcf_to_assets = float(np.clip(rng.normal(0.085, 0.045), -0.10, 0.28))
        net_debt_to_ebitda = float(np.clip(rng.normal(1.6, 1.1), -1.5, 5.5))
        deal_ev_usd_m = float(np.round(np.exp(rng.normal(6.8, 1.0)), 1))

        # Structural acquisition premium percentage formula grounded in M&A literature
        # (Baker, Pan & Wurgler 2012 exact 52-week high reference-point elasticity ~0.38)
        base_prem = SECTOR_BASE_PREMIUMS[sec]
        ref_52w_prem = min(reference_point_premium_pct(dist_52w), 90.0)
        structural_prem = (
            base_prem
            + 24.0 * sector_discount
            + 0.38 * ref_52w_prem
            + 26.0 * rev_growth_1y
            + 42.0 * fcf_to_assets
            + 12.0 * (vol_90d - 0.25)
            - 95.0 * (interest_rate_env - 0.035)
            - 2.2 * max(net_debt_to_ebitda - 1.5, 0.0)
        )
        noise = rng.normal(0.0, 5.2)
        premium_pct = float(np.clip(structural_prem + noise, 8.0, 82.0))

        pairs = PRECEDENT_ACQUIRER_COUNTERPARTIES.get(sec, [("Target Corp", "Strategic Buyer")])
        t_name, a_name = pairs[i % len(pairs)]

        rows.append(
            {
                "deal_id": f"DEAL-{1000 + i}",
                "deal_year": yr,
                "sector": sec,
                "region": reg,
                "target_name": f"{t_name} #{10 + (i // len(sectors))}",
                "acquirer_name": a_name,
                "deal_ev_usd_m": deal_ev_usd_m,
                "ev_to_ebitda": round(ev_to_ebitda, 2),
                "price_to_book": round(price_to_book, 2),
                "sector_ev_ebitda_discount": round(sector_discount, 4),
                "stock_volatility_90d": round(vol_90d, 4),
                "dist_from_52w_high": round(dist_52w, 4),
                "interest_rate_env": round(interest_rate_env, 4),
                "rev_growth_1y": round(rev_growth_1y, 4),
                "fcf_to_assets": round(fcf_to_assets, 4),
                "net_debt_to_ebitda": round(net_debt_to_ebitda, 2),
                "acquisition_premium_pct": round(premium_pct, 2),
            }
        )

    return pd.DataFrame(rows)


class AcquisitionPremiumEngine:
    """
    Trains Quantile Gradient Boosting Regressors (q15, q50, q85) and Expected-Value GBDT
    on the calibrated historical M&A precedent transaction table to recommend offer price bands
    and simulate the target board's acceptance probability curve.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.q15_model = GradientBoostingRegressor(
            loss="quantile", alpha=0.15, n_estimators=110, max_depth=3, learning_rate=0.05, random_state=random_state
        )
        self.q50_model = GradientBoostingRegressor(
            loss="quantile", alpha=0.50, n_estimators=110, max_depth=3, learning_rate=0.05, random_state=random_state
        )
        self.q85_model = GradientBoostingRegressor(
            loss="quantile", alpha=0.85, n_estimators=110, max_depth=3, learning_rate=0.05, random_state=random_state
        )
        self.mean_model = GradientBoostingRegressor(
            loss="squared_error", n_estimators=110, max_depth=3, learning_rate=0.05, random_state=random_state
        )
        self.precedent_df_: pd.DataFrame = pd.DataFrame()
        self.feature_medians_: Dict[str, float] = {}
        self.conformal_margin_: float = 0.0
        self._is_fitted = False

    def fit(self, precedent_df: Optional[pd.DataFrame] = None) -> "AcquisitionPremiumEngine":
        df = precedent_df if precedent_df is not None else generate_precedent_transactions_dataset(random_state=self.random_state)
        self.precedent_df_ = df.copy()
        X = df[PREMIUM_FEATURE_COLS]
        y = df["acquisition_premium_pct"].values

        self.feature_medians_ = X.median().to_dict()
        # Split Conformalized Quantile Regression (Romano, Patterson & Candes, 2019)
        # preserves finite-sample marginal coverage for the nominal 70% [q15, q85] band.
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state
        )
        self.q15_model.fit(X_tr, y_tr)
        self.q50_model.fit(X_tr, y_tr)
        self.q85_model.fit(X_tr, y_tr)
        self.mean_model.fit(X_tr, y_tr)

        cal_q = np.sort(
            np.column_stack(
                [
                    self.q15_model.predict(X_cal),
                    self.q50_model.predict(X_cal),
                    self.q85_model.predict(X_cal),
                ]
            ),
            axis=1,
        )
        scores = np.maximum(cal_q[:, 0] - y_cal, y_cal - cal_q[:, 2])
        n_cal = len(scores)
        q_level = min(np.ceil(0.70 * (n_cal + 1)) / max(n_cal, 1), 1.0)
        self.conformal_margin_ = float(np.quantile(scores, q_level))
        self._is_fitted = True
        return self

    def predict_premium_band(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict non-crossing (Chernozhukov, Fernandez-Val & Galichon, 2010) and split-conformal
        calibrated (Romano et al., 2019) [low, median, high] acquisition premium percentages.
        """
        if not self._is_fitted:
            self.fit()
        X = X_df[PREMIUM_FEATURE_COLS]
        q_sorted = np.sort(
            np.column_stack(
                [
                    self.q15_model.predict(X),
                    self.q50_model.predict(X),
                    self.q85_model.predict(X),
                ]
            ),
            axis=1,
        )
        mean_pred = self.mean_model.predict(X)
        med = 0.55 * q_sorted[:, 1] + 0.45 * mean_pred
        lo = q_sorted[:, 0] - self.conformal_margin_
        hi = q_sorted[:, 2] + self.conformal_margin_
        rearranged = np.sort(np.column_stack([lo, med, hi]), axis=1)
        rearranged = np.clip(rearranged, 4.0, 90.0)
        return rearranged[:, 0], rearranged[:, 1], rearranged[:, 2]

    def recommend_offer(
        self,
        target: Dict[str, Any],
        interest_rate_env: float = 0.048,
        lang: str = "tr",
    ) -> Dict[str, Any]:
        if not self._is_fitted:
            self.fit()

        en = is_en_lang(lang)
        sec_map = get_sector_map(lang)

        x_row = pd.DataFrame(
            [
                {
                    "ev_to_ebitda": float(target["ev_to_ebitda"]),
                    "price_to_book": float(target["price_to_book"]),
                    "sector_ev_ebitda_discount": float(target["sector_ev_ebitda_discount"]),
                    "stock_volatility_90d": float(target.get("stock_volatility_90d", 0.28)),
                    "dist_from_52w_high": float(target.get("dist_from_52w_high", -0.18)),
                    "interest_rate_env": float(interest_rate_env),
                    "rev_growth_1y": float(target["rev_growth_1y"]),
                    "fcf_to_assets": float(target["fcf_to_assets"]),
                    "net_debt_to_ebitda": float(target["net_debt_to_ebitda"]),
                }
            ]
        )

        lo_arr, med_arr, hi_arr = self.predict_premium_band(x_row)
        min_prem = float(lo_arr[0])
        med_prem = float(med_arr[0])
        max_prem = float(hi_arr[0])
        # Guarantee strict visual separation for the 3-point offer price band when margin collapses
        if min_prem >= med_prem - 1.0:
            min_prem = max(4.0, med_prem - 4.0)
        if max_prem <= med_prem + 1.0:
            max_prem = min(90.0, med_prem + 4.5)

        # Equity-to-Enterprise Value Offer Bridge (Premium applies strictly to Equity / Market Cap!)
        base_mcap = max(float(target["market_cap"]), 10.0)
        total_debt = float(target["total_debt"])
        cash = float(target["cash_and_equivalents"])
        net_debt = total_debt - cash
        ebitda = max(float(target["ebitda"]), 1.0)

        eq_min = round(base_mcap * (1.0 + min_prem / 100.0), 1)
        eq_med = round(base_mcap * (1.0 + med_prem / 100.0), 1)
        eq_max = round(base_mcap * (1.0 + max_prem / 100.0), 1)

        ev_min = round(max(eq_min + net_debt, ebitda * 2.0), 1)
        ev_med = round(max(eq_med + net_debt, ebitda * 2.2), 1)
        ev_max = round(max(eq_max + net_debt, ebitda * 2.5), 1)

        # Board Acceptance Probability Curve across offer premiums [0% .. 75%]
        # Incorporates exact Baker, Pan & Wurgler (2012) 52-week high reference-point kink.
        spread = max((max_prem - min_prem) * 0.34, 4.2)
        anchor_52w_prem = reference_point_premium_pct(float(target.get("dist_from_52w_high", -0.18)))
        inflection_prem = 0.72 * med_prem + 0.28 * max(min(anchor_52w_prem, 75.0), 15.0)

        acceptance_curve = []
        for p_grid in range(0, 76, 5):
            prob_accept = board_acceptance_probability(
                premium_pct=float(p_grid),
                inflection=inflection_prem,
                spread=spread,
                reference_premium=anchor_52w_prem,
            )
            offer_eq = round(base_mcap * (1.0 + p_grid / 100.0), 1)
            offer_ev = round(max(offer_eq + net_debt, ebitda * 2.0), 1)
            acceptance_curve.append(
                {
                    "premium_pct": float(p_grid),
                    "offer_equity_usd_m": offer_eq,
                    "offer_ev_usd_m": offer_ev,
                    "implied_ev_ebitda": round(offer_ev / ebitda, 2),
                    "acceptance_probability_pct": round(prob_accept, 1),
                }
            )

        # Comparable Precedent Transactions in the target's sector
        sec_code = str(target["sector_code"])
        sec_deals = self.precedent_df_[self.precedent_df_["sector"] == sec_code].copy()
        sec_deals["ev_dist"] = np.abs(sec_deals["ev_to_ebitda"] - float(target["ev_to_ebitda"]))
        closest_deals = sec_deals.sort_values("ev_dist").head(6)

        precedent_list = []
        for _, d_row in closest_deals.iterrows():
            # Exact accounting identity: EV_deal / EBITDA = EV_pre / EBITDA + (prem/100) * (Equity_pre / EBITDA)
            pre_ev_mult = float(d_row["ev_to_ebitda"])
            eq_to_ebitda = max(pre_ev_mult - float(d_row["net_debt_to_ebitda"]), 1.0)
            deal_ev_mult = round(
                pre_ev_mult + (float(d_row["acquisition_premium_pct"]) / 100.0) * eq_to_ebitda, 2
            )
            precedent_list.append(
                {
                    "deal_id": str(d_row["deal_id"]),
                    "deal_year": int(d_row["deal_year"]),
                    "target_name": sanitize_display_text(str(d_row["target_name"])),
                    "acquirer_name": sanitize_display_text(str(d_row["acquirer_name"])),
                    "sector": sec_map.get(sec_code, sec_code),
                    "deal_ev_usd_m": float(d_row["deal_ev_usd_m"]),
                    "pre_deal_ev_ebitda": float(d_row["ev_to_ebitda"]),
                    "transaction_ev_ebitda": deal_ev_mult,
                    "premium_paid_pct": float(d_row["acquisition_premium_pct"]),
                }
            )

        # Feature attribution breakdown for the recommended premium
        driver_Metadata = [
            ("sector_ev_ebitda_discount", 24.0, "Bölgesel Sektör İskontosu", "Regional Sector Valuation Discount"),
            ("dist_from_52w_high", -18.0, "52 Haftalık Zirveye Uzaklık Çıpası", "52-Week High Reference Anchor"),
            ("rev_growth_1y", 26.0, "Yıllık Ciro Büyüme İvmesi", "Annual Revenue Growth Momentum"),
            ("fcf_to_assets", 42.0, "Serbest Nakit Akışı Üretim Gücü", "Free Cash Flow Conversion"),
            ("stock_volatility_90d", 12.0, "90 Günlük Hisse Volatilitesi", "90-Day Equity Volatility"),
            ("interest_rate_env", -95.0, "Makro Faiz Ortamı Etkisi", "Macro Interest Rate Environment"),
            ("net_debt_to_ebitda", -2.2, "Mevcut Bilanço Kaldıracı", "Existing Balance Sheet Leverage"),
        ]

        premium_drivers = []
        for col_name, weight_coef, lbl_tr, lbl_en in driver_Metadata:
            val = float(x_row.loc[0, col_name])
            med_ref = float(self.feature_medians_.get(col_name, 0.0))
            impact_pct_pts = round((val - med_ref) * weight_coef, 2)
            premium_drivers.append(
                {
                    "feature": col_name,
                    "label": lbl_en if en else lbl_tr,
                    "impact_pct_pts": impact_pct_pts,
                    "direction": ("Positive" if impact_pct_pts >= 0 else "Negative")
                    if en
                    else ("Pozitif" if impact_pct_pts >= 0 else "Negatif"),
                }
            )
        premium_drivers.sort(key=lambda d: abs(d["impact_pct_pts"]), reverse=True)

        return {
            "target_ticker": target["ticker"],
            "target_name": target["company_name"],
            "sector": sec_map.get(sec_code, sec_code),
            "standalone_market_cap_usd_m": round(base_mcap, 1),
            "standalone_ev_usd_m": round(float(target["enterprise_value"]), 1),
            "standalone_ev_ebitda": round(float(target["ev_to_ebitda"]), 2),
            "min_premium_pct": round(min_prem, 1),
            "median_premium_pct": round(med_prem, 1),
            "max_premium_pct": round(max_prem, 1),
            "offer_equity_min_usd_m": eq_min,
            "offer_equity_median_usd_m": eq_med,
            "offer_equity_max_usd_m": eq_max,
            "offer_ev_min_usd_m": ev_min,
            "offer_ev_median_usd_m": ev_med,
            "offer_ev_max_usd_m": ev_max,
            "implied_tx_ev_ebitda_min": round(ev_min / ebitda, 2),
            "implied_tx_ev_ebitda_median": round(ev_med / ebitda, 2),
            "implied_tx_ev_ebitda_max": round(ev_max / ebitda, 2),
            "board_acceptance_at_median_pct": round(
                float(np.interp(med_prem, [c["premium_pct"] for c in acceptance_curve], [c["acceptance_probability_pct"] for c in acceptance_curve])),
                1,
            ),
            "acceptance_curve": acceptance_curve,
            "precedent_deals": precedent_list,
            "premium_drivers": premium_drivers,
        }
