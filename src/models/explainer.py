"""SHAP-based Explainable AI (XAI) engine for M&A target deal rationale generation."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap

from src.config import ALL_RAW_FEATURES, FEATURE_LABELS_EN, FEATURE_LABELS_TR


def _format_feature_value(feat: str, val: float) -> str:
    if feat in ("ev_to_ebitda", "price_to_book", "price_to_sales", "net_debt_to_ebitda", "debt_to_equity", "current_ratio", "interest_coverage", "asset_turnover"):
        return f"{val:.2f}x"
    if feat in (
        "sector_ev_ebitda_discount",
        "sector_pb_discount",
        "rev_growth_1y",
        "rev_growth_3y_cagr",
        "fcf_to_assets",
        "cash_to_assets",
        "capex_to_revenue",
        "ebitda_margin",
        "roa",
        "roe",
        "stock_return_12m",
        "stock_volatility_12m",
        "margin_momentum_4q",
    ):
        return f"%{val * 100:.1f}"
    return f"{val:.2f}"


def _build_turkish_driver_commentary(feat: str, val: float, shap_val: float, is_positive: bool) -> str:
    label = FEATURE_LABELS_TR.get(feat, feat)
    fmt_val = _format_feature_value(feat, val)
    direction = "artırıyor" if is_positive else "sınırlıyor"

    custom_templates = {
        "sector_ev_ebitda_discount": (
            f"Bölgesel sektör medyanına kıyasla {fmt_val} iskonto oranı cazip değerleme zemini oluşturuyor"
            if is_positive
            else f"Bölgesel sektör medyanına kıyasla {fmt_val} seviyesindeki çarpan farkı satın alma primini daraltıyor"
        ),
        "ev_to_ebitda": (
            f"{fmt_val} seviyesindeki {label} çarpanı stratejik ve finansal alıcılar için değerleme alanı sağlıyor"
            if is_positive
            else f"{fmt_val} seviyesindeki yüksek {label} çarpanı yatırım geri dönüş süresini uzatıyor"
        ),
        "growth_resource_mismatch": (
            f"{fmt_val} düzeyindeki {label} sermaye ve büyüme sinerjisi gereksinimini öne çıkarıyor"
            if is_positive
            else f"{fmt_val} düzeyindeki dengeli büyüme ve likidite yapısı birleşme baskısını azaltıyor"
        ),
        "fcf_to_assets": (
            f"{fmt_val} seviyesindeki güçlü {label} satın alma sonrası borç servis kapasitesini destekliyor"
            if is_positive
            else f"{fmt_val} seviyesindeki sınırlı {label} kaldıraçlı finansman kapasitesini kısıtlıyor"
        ),
        "net_debt_to_ebitda": (
            f"{fmt_val} düzeyindeki kontrollü {label} bilançoda ilave borçlanma kapasitesi bırakıyor"
            if is_positive
            else f"{fmt_val} düzeyindeki yüksek {label} mevcut borç yükü nedeniyle bilanço riskini artırıyor"
        ),
        "stock_return_12m": (
            f"{fmt_val} seviyesindeki yıllık hisse performansı piyasa değerlemesinde alım fırsatı sunuyor"
            if is_positive
            else f"{fmt_val} seviyesindeki güçlü hisse yükselişi kontrol primi maliyetini yukarı çekiyor"
        ),
    }

    if feat in custom_templates:
        return custom_templates[feat]
    return f"{fmt_val} seviyesindeki {label} satın alınma olasılığını {direction} — SHAP katkısı {shap_val:+.3f}"


def _build_english_driver_commentary(feat: str, val: float, shap_val: float, is_positive: bool) -> str:
    label = FEATURE_LABELS_EN.get(feat, feat)
    fmt_val = _format_feature_value(feat, val)
    direction = "increases" if is_positive else "constrains"

    custom_templates = {
        "sector_ev_ebitda_discount": (
            f"Valuation discount of {fmt_val} relative to the regional sector median creates attractive acquisition headroom"
            if is_positive
            else f"Valuation differential of {fmt_val} versus the regional sector median compresses acquisition premium capacity"
        ),
        "ev_to_ebitda": (
            f"{label} multiple at {fmt_val} supports strategic and financial buyer return thresholds"
            if is_positive
            else f"Elevated {label} multiple at {fmt_val} extends payback duration for prospective acquirers"
        ),
        "growth_resource_mismatch": (
            f"{label} at {fmt_val} indicates structural capital-to-growth imbalance favoring M&A synergy"
            if is_positive
            else f"Balanced growth and liquidity profile at {fmt_val} reduces external consolidation pressure"
        ),
        "fcf_to_assets": (
            f"Strong {label} at {fmt_val} enhances post-acquisition debt service capacity"
            if is_positive
            else f"Constrained {label} at {fmt_val} limits leveraged buyout financing flexibility"
        ),
        "net_debt_to_ebitda": (
            f"Contained {label} at {fmt_val} preserves unused debt capacity on the balance sheet"
            if is_positive
            else f"Elevated {label} at {fmt_val} increases balance sheet risk for prospective buyers"
        ),
        "stock_return_12m": (
            f"Trailing equity performance of {fmt_val} creates an opportunistic valuation entry point"
            if is_positive
            else f"Strong trailing equity rally of {fmt_val} raises control premium acquisition cost"
        ),
    }

    if feat in custom_templates:
        return custom_templates[feat]
    return f"{label} at {fmt_val} {direction} acquisition probability — SHAP contribution {shap_val:+.3f}"


class MAExplainer:
    """
    Wraps SHAP TreeExplainer / LinearExplainer to extract company-level Top 3 M&A Bullish Drivers
    and Top 3 M&A Deterrents with Turkish Investment Banking narrative summaries.
    """

    def __init__(
        self,
        model: Any,
        feature_cols: List[str] = ALL_RAW_FEATURES,
        background_data: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.background_data = background_data
        self._explainer: Any = None

    def _init_explainer(self, X_sample: pd.DataFrame) -> None:
        if self._explainer is not None or hasattr(self.model, "get_booster"):
            return
        bg = self.background_data[self.feature_cols] if self.background_data is not None else X_sample[self.feature_cols]
        if hasattr(self.model, "named_steps"):
            # Scikit-learn Pipeline (Logistic Regression)
            scaler = self.model.named_steps["scaler"]
            clf = self.model.named_steps["clf"]
            bg_scaled = scaler.transform(bg)
            self._explainer = shap.LinearExplainer(clf, bg_scaled)
        else:
            self._explainer = shap.TreeExplainer(self.model)

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        X_sub = X[self.feature_cols]
        if hasattr(self.model, "get_booster"):
            import xgboost as xgb

            booster = self.model.get_booster()
            dmat = xgb.DMatrix(X_sub, feature_names=list(self.feature_cols))
            contribs = booster.predict(dmat, pred_contribs=True)
            return np.asarray(contribs[:, :-1], dtype=float)

        self._init_explainer(X_sub)

        if hasattr(self.model, "named_steps"):
            X_input = self.model.named_steps["scaler"].transform(X_sub)
            raw_shap = self._explainer.shap_values(X_input)
        else:
            raw_shap = self._explainer.shap_values(X_sub)

        if isinstance(raw_shap, list):
            # Binary classification list [class_0, class_1]
            arr = np.asarray(raw_shap[1], dtype=float)
        else:
            arr = np.asarray(raw_shap, dtype=float)
            if arr.ndim == 3 and arr.shape[-1] == 2:
                arr = arr[:, :, 1]

        return arr

    def explain_company(self, company_row: pd.Series, shap_row: np.ndarray, lang: str = "tr") -> Dict[str, Any]:
        """Extract Top-3 positive M&A drivers and Top-3 negative deterrents for a single company in TR or EN."""
        is_en = str(lang).lower().startswith("en")
        label_map = FEATURE_LABELS_EN if is_en else FEATURE_LABELS_TR
        comment_fn = _build_english_driver_commentary if is_en else _build_turkish_driver_commentary

        shap_arr = np.asarray(shap_row, dtype=float).flatten()
        pos_order = np.argsort(-shap_arr)
        neg_order = np.argsort(shap_arr)

        top_positive = []
        for idx in pos_order[:3]:
            feat = self.feature_cols[int(idx)]
            val = float(company_row[feat])
            s_val = float(shap_arr[int(idx)])
            lbl = label_map.get(feat, feat)
            cmt = comment_fn(feat, val, s_val, is_positive=True)
            top_positive.append(
                {
                    "feature": feat,
                    "label": lbl,
                    "label_tr": lbl,
                    "value": val,
                    "formatted_value": _format_feature_value(feat, val),
                    "shap_value": round(s_val, 4),
                    "commentary": cmt,
                    "commentary_tr": cmt,
                }
            )

        top_negative = []
        for idx in neg_order[:3]:
            feat = self.feature_cols[int(idx)]
            val = float(company_row[feat])
            s_val = float(shap_arr[int(idx)])
            lbl = label_map.get(feat, feat)
            cmt = comment_fn(feat, val, s_val, is_positive=False)
            top_negative.append(
                {
                    "feature": feat,
                    "label": lbl,
                    "label_tr": lbl,
                    "value": val,
                    "formatted_value": _format_feature_value(feat, val),
                    "shap_value": round(s_val, 4),
                    "commentary": cmt,
                    "commentary_tr": cmt,
                }
            )

        primary_rationale = " · ".join(
            [f"{d['label']}: {d['formatted_value']}" for d in top_positive[:2]]
        )

        return {
            "top_positive_drivers": top_positive,
            "top_negative_deterrents": top_negative,
            "primary_rationale": primary_rationale,
            "primary_rationale_tr": primary_rationale,
            "detailed_memo_tr": top_positive[0]["commentary"],
        }

    def build_explanations_table(self, df: pd.DataFrame, shap_matrix: np.ndarray) -> pd.DataFrame:
        """Annotate a DataFrame with Top-3 positive/negative factor columns and summary rationales."""
        out = df.copy().reset_index(drop=True)
        pos_1, pos_2, pos_3 = [], [], []
        neg_1, neg_2, neg_3 = [], [], []
        rationales = []

        for i in range(len(out)):
            exp = self.explain_company(out.iloc[i], shap_matrix[i])
            p = exp["top_positive_drivers"]
            n = exp["top_negative_deterrents"]
            pos_1.append(f"{p[0]['label_tr']} ({p[0]['formatted_value']}, {p[0]['shap_value']:+.2f})")
            pos_2.append(f"{p[1]['label_tr']} ({p[1]['formatted_value']}, {p[1]['shap_value']:+.2f})")
            pos_3.append(f"{p[2]['label_tr']} ({p[2]['formatted_value']}, {p[2]['shap_value']:+.2f})")
            neg_1.append(f"{n[0]['label_tr']} ({n[0]['formatted_value']}, {n[0]['shap_value']:+.2f})")
            neg_2.append(f"{n[1]['label_tr']} ({n[1]['formatted_value']}, {n[1]['shap_value']:+.2f})")
            neg_3.append(f"{n[2]['label_tr']} ({n[2]['formatted_value']}, {n[2]['shap_value']:+.2f})")
            rationales.append(exp["primary_rationale_tr"])

        out["top_driver_1"] = pos_1
        out["top_driver_2"] = pos_2
        out["top_driver_3"] = pos_3
        out["top_deterrent_1"] = neg_1
        out["top_deterrent_2"] = neg_2
        out["top_deterrent_3"] = neg_3
        out["primary_ma_rationale"] = rationales
        return out
