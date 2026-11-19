"""Clean institutional bilingual (TR / EN) formatting and serialization utilities for the Origon REST API."""

import re
from typing import Any, Dict, List
import pandas as pd

from src.config import FEATURE_LABELS_EN, FEATURE_LABELS_TR

REGION_DISPLAY_MAP_TR: Dict[str, str] = {
    "US (NYSE/NASDAQ)": "ABD Pazarı",
    "Europe (Euronext/XETRA/LSE)": "Avrupa Pazarı",
    "Turkey (Borsa İstanbul - BIST)": "Türkiye Borsa İstanbul",
}

REGION_DISPLAY_MAP_EN: Dict[str, str] = {
    "US (NYSE/NASDAQ)": "United States Market",
    "Europe (Euronext/XETRA/LSE)": "European Market",
    "Turkey (Borsa İstanbul - BIST)": "Turkey Borsa Istanbul",
}

SECTOR_DISPLAY_MAP_TR: Dict[str, str] = {
    "Technology": "Teknoloji",
    "Healthcare": "Sağlık",
    "Industrials": "Endüstriyel",
    "Consumer Discretionary": "Tüketici Ürünleri",
    "Consumer Staples": "Temel Tüketim",
    "Energy": "Enerji",
    "Financials": "Finansal Kurumlar",
    "Materials": "Temel Maddeler",
}

SECTOR_DISPLAY_MAP_EN: Dict[str, str] = {
    "Technology": "Technology",
    "Healthcare": "Healthcare",
    "Industrials": "Industrials",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy",
    "Financials": "Financials",
    "Materials": "Materials",
}

MODEL_DISPLAY_TR: Dict[str, str] = {
    "Logistic Regression (ElasticNet L1/L2)": "Lojistik Regresyon ElasticNet",
    "XGBoost Classifier": "XGBoost Sınıflandırıcı",
    "LightGBM Classifier": "LightGBM Sınıflandırıcı",
}

MODEL_DISPLAY_EN: Dict[str, str] = {
    "Logistic Regression (ElasticNet L1/L2)": "Logistic Regression ElasticNet",
    "XGBoost Classifier": "XGBoost Classifier",
    "LightGBM Classifier": "LightGBM Classifier",
}

STRATEGY_DISPLAY_TR: Dict[str, str] = {
    "scale_pos_weight": "Sınıf Ağırlıklandırma",
    "smote": "Sentetik Azınlık Örnekleme",
    "class_weight": "Dengeli Ağırlık",
    "none": "Standart",
}

STRATEGY_DISPLAY_EN: Dict[str, str] = {
    "scale_pos_weight": "Class Cost Weighting",
    "smote": "Synthetic Minority Oversampling",
    "class_weight": "Balanced Weight",
    "none": "Unweighted",
}

CALIBRATION_DISPLAY_TR: Dict[str, str] = {
    "platt": "Platt Sigmoid Kalibrasyonu",
    "isotonic": "İzotonik Regresyon",
    "uncalibrated": "Ham Skor",
}

CALIBRATION_DISPLAY_EN: Dict[str, str] = {
    "platt": "Platt Sigmoid Calibration",
    "isotonic": "Isotonic Regression",
    "uncalibrated": "Uncalibrated Score",
}

# Backwards-compatible aliases
REGION_DISPLAY_MAP = REGION_DISPLAY_MAP_TR
SECTOR_DISPLAY_MAP = SECTOR_DISPLAY_MAP_TR
MODEL_DISPLAY_CLEAN = MODEL_DISPLAY_TR
STRATEGY_DISPLAY_CLEAN = STRATEGY_DISPLAY_TR
CALIBRATION_DISPLAY_CLEAN = CALIBRATION_DISPLAY_TR


def is_en_lang(lang: str) -> bool:
    return str(lang or "tr").lower().strip().startswith("en")


def get_feature_labels(lang: str = "tr") -> Dict[str, str]:
    return FEATURE_LABELS_EN if is_en_lang(lang) else FEATURE_LABELS_TR


def get_region_map(lang: str = "tr") -> Dict[str, str]:
    return REGION_DISPLAY_MAP_EN if is_en_lang(lang) else REGION_DISPLAY_MAP_TR


def get_sector_map(lang: str = "tr") -> Dict[str, str]:
    return SECTOR_DISPLAY_MAP_EN if is_en_lang(lang) else SECTOR_DISPLAY_MAP_TR


def get_size_buckets(lang: str = "tr") -> List[str]:
    if is_en_lang(lang):
        return ["Small Cap", "Mid Cap", "Large Cap"]
    return ["Küçük Ölçek", "Orta Ölçek", "Büyük Ölçek"]


def assign_clean_size_bucket(mcap_millions_usd: float, lang: str = "tr") -> str:
    if is_en_lang(lang):
        if mcap_millions_usd < 1000.0:
            return "Small Cap"
        if mcap_millions_usd <= 10000.0:
            return "Mid Cap"
        return "Large Cap"
    if mcap_millions_usd < 1000.0:
        return "Küçük Ölçek"
    if mcap_millions_usd <= 10000.0:
        return "Orta Ölçek"
    return "Büyük Ölçek"


def sanitize_display_text(text: str) -> str:
    """Remove parenthetical segments and quotation marks for strict institutional typography."""
    if not text:
        return ""
    cleaned = re.sub(r"\s*\([^)]*\)", "", str(text))
    cleaned = cleaned.replace('"', "").replace("'", "").strip()
    return cleaned


def clean_company_title(raw_name: str, ticker: str, region_raw: str, lang: str = "tr") -> str:
    base = sanitize_display_text(raw_name)
    if not base or base.upper() == ticker.upper():
        if "Turkey" in region_raw or ticker.endswith(".IS"):
            suffix = "Industrial Group" if is_en_lang(lang) else "Yatırım Ortaklığı"
            return f"{ticker.replace('.IS', '')} {suffix}"
        if "Europe" in region_raw:
            return f"{ticker.split('.')[0]} Holdings"
        return f"{ticker} Corporation"
    return base


def clean_model_title(raw_title: str, lang: str = "tr") -> str:
    base = re.sub(r"\s*\[[^\]]*\]", "", str(raw_title)).strip()
    m_map = MODEL_DISPLAY_EN if is_en_lang(lang) else MODEL_DISPLAY_TR
    return m_map.get(base, sanitize_display_text(base))


def translate_rationale_string(raw_rationale: str, lang: str = "tr") -> str:
    cleaned = sanitize_display_text(raw_rationale).replace(" | ", " · ")
    if not is_en_lang(lang):
        return cleaned
    out = cleaned
    for feat_key, tr_lbl in FEATURE_LABELS_TR.items():
        en_lbl = FEATURE_LABELS_EN.get(feat_key, feat_key)
        out = out.replace(tr_lbl, en_lbl)
    return out


def serialize_target_row(row: pd.Series, lang: str = "tr") -> Dict[str, Any]:
    en = is_en_lang(lang)
    reg_raw = str(row.get("region", "US (NYSE/NASDAQ)"))
    sec_raw = str(row.get("sector", "Technology"))
    mcap = float(row.get("market_cap", 0.0))
    src_raw = str(row.get("data_source", "Historical_Benchmark"))

    reg_map = get_region_map(lang)
    sec_map = get_sector_map(lang)

    if en:
        data_source_label = "Live Market Feed" if src_raw == "Live_Yahoo_Finance" else "Historical Panel"
    else:
        data_source_label = "Canlı Piyasa Akışı" if src_raw == "Live_Yahoo_Finance" else "Tarihsel Veri Havuzu"

    return {
        "rank": int(row["ma_rank"]),
        "ticker": str(row["ticker"]),
        "company_name": clean_company_title(str(row.get("company_name", "")), str(row["ticker"]), reg_raw, lang=lang),
        "region": reg_map.get(reg_raw, sanitize_display_text(reg_raw)),
        "region_code": reg_raw,
        "sector": sec_map.get(sec_raw, sanitize_display_text(sec_raw)),
        "sector_code": sec_raw,
        "size_bucket": assign_clean_size_bucket(mcap, lang=lang),
        "currency": str(row.get("currency", "USD")),
        "data_source": data_source_label,
        "is_live": src_raw == "Live_Yahoo_Finance",
        "probability_pct": round(float(row["predicted_ma_pct"]), 2),
        "market_cap_usd_m": round(mcap, 1),
        "enterprise_value_usd_m": round(float(row.get("enterprise_value", mcap)), 1),
        "ev_to_ebitda": round(float(row["ev_to_ebitda"]), 2),
        "price_to_book": round(float(row["price_to_book"]), 2),
        "price_to_sales": round(float(row["price_to_sales"]), 2),
        "sector_discount_pct": round(float(row["sector_ev_ebitda_discount"]) * 100.0, 1),
        "net_debt_to_ebitda": round(float(row["net_debt_to_ebitda"]), 2),
        "fcf_to_assets_pct": round(float(row["fcf_to_assets"]) * 100.0, 1),
        "cash_to_assets_pct": round(float(row["cash_to_assets"]) * 100.0, 1),
        "rev_growth_1y_pct": round(float(row["rev_growth_1y"]) * 100.0, 1),
        "ebitda_margin_pct": round(float(row["ebitda_margin"]) * 100.0, 1),
        "roa_pct": round(float(row["roa"]) * 100.0, 1),
        "roe_pct": round(float(row["roe"]) * 100.0, 1),
        "stock_return_12m_pct": round(float(row["stock_return_12m"]) * 100.0, 1),
        "growth_mismatch_idx": round(float(row["growth_resource_mismatch"]), 2),
        "primary_rationale": translate_rationale_string(str(row.get("primary_ma_rationale", "")), lang=lang),
    }


def build_peer_benchmarks(comp_row: pd.Series, sec_med_row: pd.Series, lang: str = "tr") -> List[Dict[str, Any]]:
    f_labels = get_feature_labels(lang)
    specs = [
        ("ev_to_ebitda", "x", 1.0),
        ("price_to_book", "x", 1.0),
        ("price_to_sales", "x", 1.0),
        ("rev_growth_1y", "%", 100.0),
        ("fcf_to_assets", "%", 100.0),
        ("cash_to_assets", "%", 100.0),
        ("net_debt_to_ebitda", "x", 1.0),
        ("debt_to_equity", "x", 1.0),
        ("current_ratio", "x", 1.0),
        ("ebitda_margin", "%", 100.0),
        ("roa", "%", 100.0),
        ("roe", "%", 100.0),
        ("stock_return_12m", "%", 100.0),
    ]
    rows: List[Dict[str, Any]] = []
    for feat_key, unit, mult in specs:
        c_val = float(comp_row[feat_key]) * mult
        s_val = float(sec_med_row[feat_key]) * mult
        diff_pct = ((c_val - s_val) / max(abs(s_val), 1e-3)) * 100.0
        z_val = float(comp_row.get(f"{feat_key}_sector_z", 0.0))

        rows.append(
            {
                "feature": feat_key,
                "label": f_labels.get(feat_key, feat_key),
                "unit": unit,
                "company_value": round(c_val, 2),
                "sector_median": round(s_val, 2),
                "company_formatted": f"{c_val:.2f}x" if unit == "x" else f"%{c_val:.1f}",
                "median_formatted": f"{s_val:.2f}x" if unit == "x" else f"%{s_val:.1f}",
                "diff_pct": round(diff_pct, 1),
                "diff_formatted": f"%{diff_pct:+.1f}",
                "sector_z": round(z_val, 2),
                "z_formatted": f"{z_val:+.2f}σ",
            }
        )
    return rows
