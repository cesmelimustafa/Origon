"""Module 2: Strategic and Financial Buyer Matching Engine using TF-IDF cosine similarity and financial capacity."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.advisory.buyer_catalog import (
    build_conglomerate_buyer_catalog,
    build_pe_sponsor_catalog,
    build_strategic_buyer_catalog,
    build_swf_buyer_catalog,
)
from src.api.schemas import get_region_map, get_sector_map, is_en_lang, sanitize_display_text
from src.data.benchmark_generator import REGION_PARAMETERS, SECTOR_PROFILES

BUYER_REGION_CODES: List[str] = [
    "US (NYSE/NASDAQ)",
    "Europe (Euronext/XETRA/LSE)",
    "Turkey (Borsa İstanbul - BIST)",
    "MENA / Global Capital",
]

VERTICAL_ADJACENCY: Dict[str, List[str]] = {
    "Technology": ["Financials", "Healthcare", "Industrials", "Consumer Discretionary"],
    "Healthcare": ["Technology", "Materials", "Consumer Staples"],
    "Industrials": ["Materials", "Technology", "Energy"],
    "Consumer Discretionary": ["Consumer Staples", "Technology"],
    "Consumer Staples": ["Consumer Discretionary", "Materials"],
    "Energy": ["Industrials", "Materials"],
    "Financials": ["Technology"],
    "Materials": ["Industrials", "Energy", "Consumer Staples"],
}


class BuyerMatchingEngine:
    """
    Ranks Strategic Acquirers and 40 SEC Form ADV Private Equity Sponsors against a target company
    across four mathematical dimensions:
    1. NLP Vector Cosine Similarity & Horizontal/Vertical Industry Fit (28%)
    2. Financial Capacity & Unused Debt Headroom / PE Ticket & Leverage Fit (30%)
    3. Historical M&A Propensity & Deal Scale Alignment (20%)
    4. Operational Synergy & Growth-Margin Complementarity (22%)
    """

    def __init__(self):
        self.pe_df = build_pe_sponsor_catalog()
        self.strat_df = build_strategic_buyer_catalog()
        self.cong_df = build_conglomerate_buyer_catalog()
        self.swf_df = build_swf_buyer_catalog()

    def _normalize_region_filter(self, buyer_regions: Optional[List[str]]) -> Optional[set]:
        if not buyer_regions:
            return None
        alias_map = {
            "us": "US (NYSE/NASDAQ)",
            "abd pazarı": "US (NYSE/NASDAQ)",
            "united states market": "US (NYSE/NASDAQ)",
            "europe": "Europe (Euronext/XETRA/LSE)",
            "avrupa pazarı": "Europe (Euronext/XETRA/LSE)",
            "european market": "Europe (Euronext/XETRA/LSE)",
            "turkey": "Turkey (Borsa İstanbul - BIST)",
            "türkiye borsa istanbul": "Turkey (Borsa İstanbul - BIST)",
            "turkey borsa istanbul": "Turkey (Borsa İstanbul - BIST)",
            "mena": "MENA / Global Capital",
            "körfez ve küresel sermaye": "MENA / Global Capital",
            "mena and global capital": "MENA / Global Capital",
        }
        resolved = set()
        for raw in buyer_regions:
            s = str(raw).strip()
            if not s:
                continue
            if s in BUYER_REGION_CODES:
                resolved.add(s)
            else:
                mapped = alias_map.get(s.lower())
                if mapped:
                    resolved.add(mapped)
        return resolved if resolved else None

    def _compute_nlp_scores(self, target_text: str, buyer_texts: List[str]) -> np.ndarray:
        corpus = [target_text] + buyer_texts
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_mat = vec.fit_transform(corpus)
        sims = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()
        # Rescale cosine similarity [0.0, 0.55] into institutional [35, 98] range
        scaled = 35.0 + np.clip(sims / 0.45, 0.0, 1.0) * 63.0
        return np.round(scaled, 1)

    def match_buyers(
        self,
        target: Dict[str, Any],
        top_k: int = 5,
        lang: str = "tr",
        buyer_regions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        en = is_en_lang(lang)
        reg_map = get_region_map(lang)
        sec_map = get_sector_map(lang)
        allowed_regions = self._normalize_region_filter(buyer_regions)

        target_sec = str(target["sector_code"])
        target_reg = str(target["region_code"])
        target_ev = max(float(target["enterprise_value"]) * 1.25, 40.0)  # Implied transaction EV with ~25% premium
        target_mcap = max(float(target["market_cap"]) * 1.25, 25.0)
        target_lev = float(target["net_debt_to_ebitda"])
        target_growth = float(target["rev_growth_1y"])
        target_margin = float(target["ebitda_margin"])
        target_ev_ebitda = max(float(target.get("ev_to_ebitda", 10.0)), 2.0)
        target_discount = float(target.get("sector_ev_ebitda_discount", 0.0))
        target_desc = str(target.get("business_description", target_sec))

        # 1. Score Strategic Acquirers
        strat_texts = [
            f"{row['sector']} {row['business_description']}" for _, row in self.strat_df.iterrows()
        ]
        strat_nlp_base = self._compute_nlp_scores(target_desc, strat_texts)

        strategic_candidates: List[Dict[str, Any]] = []
        for idx, row in self.strat_df.iterrows():
            if str(row["ticker"]).upper() == str(target.get("ticker", "")).upper():
                continue

            b_sec = str(row["sector"])
            b_reg = str(row["region"])
            if allowed_regions is not None and b_reg not in allowed_regions:
                continue

            # Horizontal / vertical boost
            if b_sec == target_sec:
                ind_boost = 18.0
            elif b_sec in VERTICAL_ADJACENCY.get(target_sec, []):
                ind_boost = 7.0
            else:
                ind_boost = -14.0
            nlp_score = float(np.clip(strat_nlp_base[idx] + ind_boost, 15.0, 99.0))

            # Hoberg & Phillips (2010) horizontal antitrust / concentration friction:
            # large horizontal combinations in concentrated markets face regulatory scrutiny.
            buyer_ebitda = max(float(row["ebitda_usd_m"]), 1.0)
            antitrust_penalty_pts = 0.0
            if b_sec == target_sec and target_ev >= 2500.0 and buyer_ebitda >= 3000.0:
                size_factor = min(np.log1p(target_ev / 2500.0), 3.0)
                reg_mult = 1.15 if b_reg == target_reg else 0.70
                antitrust_penalty_pts = round(float(np.clip(3.2 * size_factor * reg_mult, 0.0, 14.0)), 2)

            # Financial Capacity = Cash + Unused Debt Capacity
            unused_debt_turns = max(0.0, float(row["max_sector_leverage"]) - float(row["net_debt_ebitda"]))
            debt_headroom_usd_m = unused_debt_turns * buyer_ebitda
            firepower_usd_m = float(row["cash_usd_m"]) + debt_headroom_usd_m
            coverage_ratio = firepower_usd_m / max(target_ev, 1.0)
            fin_cap_score = float(np.clip(30.0 + 70.0 * (1.0 - np.exp(-0.55 * coverage_ratio)), 18.0, 99.0))

            # Historical M&A propensity & scale alignment (adjusted for horizontal antitrust friction)
            deal_freq_norm = min(float(row["deals_5y"]) / 22.0, 1.0) * 50.0
            scale_dist = abs(np.log(max(target_ev, 25.0)) - np.log(max(float(row["preferred_ev_usd_m"]), 25.0)))
            scale_fit = max(0.0, 35.0 - 11.0 * scale_dist)
            reg_bonus = 15.0 if b_reg == target_reg else 7.0
            propensity_score = float(np.clip(deal_freq_norm + scale_fit + reg_bonus - antitrust_penalty_pts, 15.0, 98.0))

            # Operational Synergy, Growth-Margin Complementarity & Rhodes-Kropf et al. (2005) Valuation Accretion
            sec_prof = SECTOR_PROFILES.get(b_sec, {"ev_ebitda": 11.0, "margin": 0.20})
            reg_scale = float(REGION_PARAMETERS.get(b_reg, {"multiple_scale": 1.0})["multiple_scale"])
            margin_rel = float(row["ebitda_margin"]) / max(float(sec_prof["margin"]), 0.05)
            buyer_ev_ebitda = float(sec_prof["ev_ebitda"]) * reg_scale * float(np.clip(0.85 + 0.25 * margin_rel, 0.75, 1.45))
            accretion_spread = float(np.clip((buyer_ev_ebitda - target_ev_ebitda) / max(target_ev_ebitda, 1.0), -0.40, 0.80))
            accretion_spread_pct = round(accretion_spread * 100.0, 1)

            margin_gap = max(float(row["ebitda_margin"]) - target_margin, -0.05)
            synergy_raw = (
                50.0
                + 115.0 * margin_gap
                + 75.0 * max(target_growth, 0.0)
                + 18.0 * accretion_spread
            )
            if b_sec == target_sec:
                synergy_raw += 10.0
            synergy_score = float(np.clip(synergy_raw, 25.0, 98.0))

            total_score = round(
                0.28 * nlp_score + 0.30 * fin_cap_score + 0.20 * propensity_score + 0.22 * synergy_score,
                1,
            )

            sec_label = sec_map.get(b_sec, b_sec)
            reg_label = reg_map.get(b_reg, sanitize_display_text(b_reg))
            if en:
                rationale = (
                    f"Strategic alignment in {sec_label} with {firepower_usd_m:,.0f}M USD acquisition firepower "
                    f"covering {min(coverage_ratio, 25.0):.1f}x target transaction value and margin expansion potential"
                )
                buyer_type_label = "Strategic Buyer"
            else:
                rationale = (
                    f"{sec_label} değer zincirinde yüksek faaliyet uyumu ve {firepower_usd_m:,.0f}M USD toplam satın alma "
                    f"kapasitesiyle hedef işlem büyüklüğünü {min(coverage_ratio, 25.0):.1f} kat karşılama gücü"
                )
                buyer_type_label = "Stratejik Alıcı"

            strategic_candidates.append(
                {
                    "buyer_id": str(row["buyer_id"]),
                    "name": sanitize_display_text(str(row["name"])),
                    "buyer_type": buyer_type_label,
                    "buyer_type_code": "strategic",
                    "region": reg_label,
                    "focus_sector": sec_label,
                    "firepower_usd_m": round(firepower_usd_m, 1),
                    "max_leverage_turns": round(float(row["max_sector_leverage"]), 1),
                    "deals_5y": int(row["deals_5y"]),
                    "total_score": total_score,
                    "accretion_spread_pct": accretion_spread_pct,
                    "antitrust_penalty_pts": antitrust_penalty_pts,
                    "breakdown": {
                        "nlp_fit_score": round(nlp_score, 1),
                        "financial_capacity_score": round(fin_cap_score, 1),
                        "deal_propensity_score": round(propensity_score, 1),
                        "synergy_score": round(synergy_score, 1),
                    },
                    "rationale": sanitize_display_text(rationale),
                }
            )

        # 2. Score 40 Private Equity Financial Sponsors
        pe_texts = [
            f"{' '.join(row['focus_sectors'])} {row['mandate']}" for _, row in self.pe_df.iterrows()
        ]
        pe_nlp_base = self._compute_nlp_scores(target_desc, pe_texts)

        financial_candidates: List[Dict[str, Any]] = []
        required_sponsor_equity = max(target_ev * 0.42, 20.0)

        for idx, row in self.pe_df.iterrows():
            focus_list: List[str] = list(row["focus_sectors"])
            b_reg = str(row["region"])
            if allowed_regions is not None and b_reg not in allowed_regions:
                continue
            sector_match = target_sec in focus_list

            nlp_score = float(
                np.clip(pe_nlp_base[idx] + (16.0 if sector_match else -12.0), 18.0, 98.0)
            )

            # PE Ticket & Leverage Fit
            min_t = float(row["min_ticket_usd_m"])
            max_t = float(row["max_ticket_usd_m"])
            if min_t <= required_sponsor_equity <= max_t:
                ticket_fit = 55.0
            else:
                dist_t = min(
                    abs(np.log(required_sponsor_equity / max(min_t, 1.0))),
                    abs(np.log(required_sponsor_equity / max(max_t, 1.0))),
                )
                ticket_fit = max(12.0, 52.0 - 22.0 * dist_t)

            lev_headroom = max(0.0, float(row["max_leverage_turns"]) - max(target_lev, 0.0))
            lev_fit = min(lev_headroom * 9.5, 43.0)
            fin_cap_score = float(np.clip(ticket_fit + lev_fit, 20.0, 99.0))

            # Deal propensity & regional alignment
            freq_score = min(float(row["deals_5y"]) / 45.0, 1.0) * 52.0
            reg_score = 32.0 if b_reg == target_reg else 18.0
            sec_bonus = 14.0 if sector_match else 0.0
            propensity_score = float(np.clip(freq_score + reg_score + sec_bonus, 22.0, 98.0))

            # LBO Cash Conversion, Growth & Kaplan-Stromberg (2009) Entry Multiple Arbitrage
            fcf_rat = float(target.get("fcf_to_assets", 0.08))
            entry_arb = float(np.clip(target_discount, -0.30, 0.50))
            synergy_score = float(
                np.clip(
                    46.0
                    + 170.0 * max(fcf_rat, 0.0)
                    + 70.0 * max(target_growth, 0.0)
                    + 22.0 * entry_arb
                    + (10.0 if sector_match else 0.0),
                    25.0,
                    97.0,
                )
            )

            total_score = round(
                0.28 * nlp_score + 0.30 * fin_cap_score + 0.20 * propensity_score + 0.22 * synergy_score,
                1,
            )

            reg_label = reg_map.get(b_reg, sanitize_display_text(b_reg))
            primary_focus = sec_map.get(target_sec if sector_match else focus_list[0], focus_list[0])
            aum_b = float(row["aum_usd_b"])

            if en:
                rationale = (
                    f"Private equity mandate covering {primary_focus} with {aum_b:.1f}B USD AUM and equity ticket window "
                    f"of {min_t:.0f}M to {max_t:.0f}M USD supporting {float(row['max_leverage_turns']):.1f}x LBO leverage"
                )
                buyer_type_label = "Financial Sponsor"
            else:
                rationale = (
                    f"{primary_focus} odaklı {aum_b:.1f} Milyar USD fon büyüklüğü ve {min_t:.0f}M ile {max_t:.0f}M USD "
                    f"özsermaye bilet aralığıyla {float(row['max_leverage_turns']):.1f}x kaldıraçlı satın alma uyumu"
                )
                buyer_type_label = "Finansal Fon Alıcısı"

            financial_candidates.append(
                {
                    "buyer_id": str(row["buyer_id"]),
                    "name": sanitize_display_text(str(row["name"])),
                    "buyer_type": buyer_type_label,
                    "buyer_type_code": "financial",
                    "region": reg_label,
                    "focus_sector": primary_focus,
                    "firepower_usd_m": round(max_t * 2.2, 1),
                    "max_leverage_turns": round(float(row["max_leverage_turns"]), 1),
                    "deals_5y": int(row["deals_5y"]),
                    "total_score": total_score,
                    "accretion_spread_pct": round(entry_arb * 100.0, 1),
                    "antitrust_penalty_pts": 0.0,
                    "breakdown": {
                        "nlp_fit_score": round(nlp_score, 1),
                        "financial_capacity_score": round(fin_cap_score, 1),
                        "deal_propensity_score": round(propensity_score, 1),
                        "synergy_score": round(synergy_score, 1),
                    },
                    "rationale": sanitize_display_text(rationale),
                }
            )

        # 3. Score Diversified Holdings & Conglomerates (multi-sector portfolio acquirers)
        cong_texts = [
            f"{' '.join(row['focus_sectors'])} {row['business_description']}" for _, row in self.cong_df.iterrows()
        ]
        cong_nlp_base = self._compute_nlp_scores(target_desc, cong_texts)

        conglomerate_candidates: List[Dict[str, Any]] = []
        for idx, row in self.cong_df.iterrows():
            if str(row["ticker"]).upper() == str(target.get("ticker", "")).upper():
                continue
            b_reg = str(row["region"])
            if allowed_regions is not None and b_reg not in allowed_regions:
                continue

            focus_list = list(row["focus_sectors"])
            sector_match = target_sec in focus_list
            nlp_score = float(np.clip(cong_nlp_base[idx] + (15.0 if sector_match else -8.0), 18.0, 98.0))

            buyer_ebitda = max(float(row["ebitda_usd_m"]), 1.0)
            unused_debt_turns = max(0.0, float(row["max_sector_leverage"]) - float(row["net_debt_ebitda"]))
            firepower_usd_m = float(row["cash_usd_m"]) + unused_debt_turns * buyer_ebitda
            coverage_ratio = firepower_usd_m / max(target_ev, 1.0)
            fin_cap_score = float(np.clip(32.0 + 68.0 * (1.0 - np.exp(-0.52 * coverage_ratio)), 20.0, 99.0))

            # Conglomerates have strong domestic/regional platform consolidation advantage and low antitrust friction
            deal_freq_norm = min(float(row["deals_5y"]) / 16.0, 1.0) * 44.0
            scale_dist = abs(np.log(max(target_ev, 25.0)) - np.log(max(float(row["preferred_ev_usd_m"]), 25.0)))
            scale_fit = max(0.0, 32.0 - 9.5 * scale_dist)
            reg_bonus = 22.0 if b_reg == target_reg else 9.0
            propensity_score = float(np.clip(deal_freq_norm + scale_fit + reg_bonus, 20.0, 98.0))

            margin_gap = max(float(row["ebitda_margin"]) - target_margin, -0.05)
            entry_arb = float(np.clip(target_discount, -0.25, 0.45))
            synergy_score = float(
                np.clip(
                    50.0
                    + 95.0 * margin_gap
                    + 80.0 * max(target_growth, 0.0)
                    + 20.0 * entry_arb
                    + (9.0 if sector_match else 0.0),
                    25.0,
                    97.0,
                )
            )
            total_score = round(
                0.26 * nlp_score + 0.30 * fin_cap_score + 0.22 * propensity_score + 0.22 * synergy_score,
                1,
            )

            reg_label = reg_map.get(b_reg, sanitize_display_text(b_reg))
            primary_focus = sec_map.get(target_sec if sector_match else focus_list[0], focus_list[0])
            if en:
                rationale = (
                    f"Diversified holding group active in {primary_focus} with {firepower_usd_m:,.0f}M USD group balance sheet "
                    f"capacity covering {min(coverage_ratio, 25.0):.1f}x deal value and portfolio diversification fit"
                )
                buyer_type_label = "Diversified Holding"
            else:
                rationale = (
                    f"{primary_focus} kolunda aktif çok sektörlü holding yapısı ve {firepower_usd_m:,.0f}M USD grup bilanço "
                    f"gücüyle hedef işlem değerini {min(coverage_ratio, 25.0):.1f} kat karşılama kapasitesi"
                )
                buyer_type_label = "Holding ve Konglomera"

            conglomerate_candidates.append(
                {
                    "buyer_id": str(row["buyer_id"]),
                    "name": sanitize_display_text(str(row["name"])),
                    "buyer_type": buyer_type_label,
                    "buyer_type_code": "conglomerate",
                    "region": reg_label,
                    "region_code": b_reg,
                    "focus_sector": primary_focus,
                    "firepower_usd_m": round(firepower_usd_m, 1),
                    "max_leverage_turns": round(float(row["max_sector_leverage"]), 1),
                    "deals_5y": int(row["deals_5y"]),
                    "total_score": total_score,
                    "accretion_spread_pct": round(entry_arb * 100.0, 1),
                    "antitrust_penalty_pts": 0.0,
                    "breakdown": {
                        "nlp_fit_score": round(nlp_score, 1),
                        "financial_capacity_score": round(fin_cap_score, 1),
                        "deal_propensity_score": round(propensity_score, 1),
                        "synergy_score": round(synergy_score, 1),
                    },
                    "rationale": sanitize_display_text(rationale),
                }
            )

        # 4. Score Sovereign Wealth Funds & Patient Institutional Capital
        swf_texts = [
            f"{' '.join(row['focus_sectors'])} {row['mandate']}" for _, row in self.swf_df.iterrows()
        ]
        swf_nlp_base = self._compute_nlp_scores(target_desc, swf_texts)

        swf_candidates: List[Dict[str, Any]] = []
        for idx, row in self.swf_df.iterrows():
            b_reg = str(row["region"])
            if allowed_regions is not None and b_reg not in allowed_regions:
                continue

            focus_list = list(row["focus_sectors"])
            corridor_list = list(row["corridor_regions"])
            sector_match = target_sec in focus_list
            corridor_match = target_reg in corridor_list or b_reg == target_reg

            nlp_score = float(np.clip(swf_nlp_base[idx] + (15.0 if sector_match else -10.0), 18.0, 98.0))

            # Patient capital invests primarily with direct equity without heavy LBO leverage reliance
            min_t = float(row["min_ticket_usd_m"])
            max_t = float(row["max_ticket_usd_m"])
            direct_ticket = max(target_ev * 0.65, 30.0)
            if min_t <= direct_ticket <= max_t:
                ticket_fit = 62.0
            else:
                dist_t = min(
                    abs(np.log(direct_ticket / max(min_t, 1.0))),
                    abs(np.log(direct_ticket / max(max_t, 1.0))),
                )
                ticket_fit = max(14.0, 58.0 - 20.0 * dist_t)

            aum_b = float(row["aum_usd_b"])
            depth_score = min(np.log1p(aum_b) * 6.5, 36.0)
            fin_cap_score = float(np.clip(ticket_fit + depth_score, 22.0, 99.0))

            freq_score = min(float(row["deals_5y"]) / 40.0, 1.0) * 46.0
            corridor_bonus = 34.0 if corridor_match else 12.0
            sec_bonus = 16.0 if sector_match else 0.0
            propensity_score = float(np.clip(freq_score + corridor_bonus + sec_bonus, 20.0, 98.0))

            fcf_rat = float(target.get("fcf_to_assets", 0.08))
            entry_arb = float(np.clip(target_discount, -0.25, 0.45))
            synergy_score = float(
                np.clip(
                    48.0
                    + 135.0 * max(fcf_rat, 0.0)
                    + 85.0 * max(target_growth, 0.0)
                    + 18.0 * entry_arb
                    + (10.0 if sector_match else 0.0),
                    25.0,
                    97.0,
                )
            )
            total_score = round(
                0.26 * nlp_score + 0.30 * fin_cap_score + 0.24 * propensity_score + 0.20 * synergy_score,
                1,
            )

            reg_label = reg_map.get(b_reg, sanitize_display_text(b_reg))
            primary_focus = sec_map.get(target_sec if sector_match else focus_list[0], focus_list[0])
            if en:
                rationale = (
                    f"Patient sovereign capital mandate in {primary_focus} with {aum_b:.0f}B USD AUM and direct equity "
                    f"ticket range of {min_t:.0f}M to {max_t:.0f}M USD without short term exit pressure"
                )
                buyer_type_label = "Sovereign Wealth Fund"
            else:
                rationale = (
                    f"{primary_focus} odaklı {aum_b:.0f} Milyar USD sabırlı sermaye büyüklüğü ve {min_t:.0f}M ile {max_t:.0f}M USD "
                    f"doğrudan özsermaye biletiyle kısa vadeli çıkış baskısı olmadan uzun vadeli ortaklık uyumu"
                )
                buyer_type_label = "Devlet Varlık Fonu"

            swf_candidates.append(
                {
                    "buyer_id": str(row["buyer_id"]),
                    "name": sanitize_display_text(str(row["name"])),
                    "buyer_type": buyer_type_label,
                    "buyer_type_code": "swf",
                    "region": reg_label,
                    "region_code": b_reg,
                    "focus_sector": primary_focus,
                    "firepower_usd_m": round(max_t * 1.5, 1),
                    "max_leverage_turns": 2.5,
                    "deals_5y": int(row["deals_5y"]),
                    "total_score": total_score,
                    "accretion_spread_pct": round(entry_arb * 100.0, 1),
                    "antitrust_penalty_pts": 0.0,
                    "breakdown": {
                        "nlp_fit_score": round(nlp_score, 1),
                        "financial_capacity_score": round(fin_cap_score, 1),
                        "deal_propensity_score": round(propensity_score, 1),
                        "synergy_score": round(synergy_score, 1),
                    },
                    "rationale": sanitize_display_text(rationale),
                }
            )

        strategic_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        financial_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        conglomerate_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        swf_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        all_candidates = sorted(
            strategic_candidates + financial_candidates + conglomerate_candidates + swf_candidates,
            key=lambda x: x["total_score"],
            reverse=True,
        )

        for rank_idx, item in enumerate(all_candidates, start=1):
            item["match_rank"] = rank_idx

        available_regions = [
            {"code": code, "label": reg_map.get(code, sanitize_display_text(code))}
            for code in BUYER_REGION_CODES
        ]
        applied_codes = sorted(list(allowed_regions)) if allowed_regions else list(BUYER_REGION_CODES)

        return {
            "target_ticker": target["ticker"],
            "target_name": target["company_name"],
            "implied_deal_ev_usd_m": round(target_ev, 1),
            "required_sponsor_equity_usd_m": round(required_sponsor_equity, 1),
            "strategic_pool_count": len(strategic_candidates),
            "financial_pool_count": len(financial_candidates),
            "conglomerate_pool_count": len(conglomerate_candidates),
            "swf_pool_count": len(swf_candidates),
            "available_buyer_regions": available_regions,
            "applied_buyer_regions": applied_codes,
            "top_overall": all_candidates[:top_k],
            "top_strategic": strategic_candidates[:top_k],
            "top_financial": financial_candidates[:top_k],
            "top_conglomerate": conglomerate_candidates[:top_k],
            "top_swf": swf_candidates[:top_k],
        }
