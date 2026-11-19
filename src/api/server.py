"""ASGI REST service layer and static web platform server for the Origon M&A Target Prediction Platform."""

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_origon_cache"))

import joblib
import pandas as pd
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.api.schemas import (
    CALIBRATION_DISPLAY_EN,
    CALIBRATION_DISPLAY_TR,
    MODEL_DISPLAY_EN,
    MODEL_DISPLAY_TR,
    STRATEGY_DISPLAY_EN,
    STRATEGY_DISPLAY_TR,
    assign_clean_size_bucket,
    build_peer_benchmarks,
    clean_model_title,
    get_feature_labels,
    get_region_map,
    get_sector_map,
    get_size_buckets,
    is_en_lang,
    sanitize_display_text,
    serialize_target_row,
)
from src.config import ALL_RAW_FEATURES, ARTIFACTS_DIR, PROJECT_ROOT
from src.models.explainer import MAExplainer
from src.pipeline import run_ma_pipeline

WEB_DIR = PROJECT_ROOT / "web"


class OrigonServiceState:
    """In-memory artifact store with on-demand reload capability."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.ranked_df: pd.DataFrame = pd.DataFrame()
        self.comp_df: pd.DataFrame = pd.DataFrame()
        self.lift_df: pd.DataFrame = pd.DataFrame()
        self.rel_df: pd.DataFrame = pd.DataFrame()
        self.sec_df: pd.DataFrame = pd.DataFrame()
        self.bundle: Dict[str, Any] = {}
        self.last_updated_iso: str = ""
        self.load_or_initialize(force_retrain=False, use_live=False)

    def load_or_initialize(self, force_retrain: bool = False, use_live: bool = True) -> None:
        ranked_path = self.artifacts_dir / "ranked_targets.parquet"
        comp_path = self.artifacts_dir / "model_comparison.csv"
        lift_path = self.artifacts_dir / "decile_lift.csv"
        rel_path = self.artifacts_dir / "reliability_curve.csv"
        sec_path = self.artifacts_dir / "sector_medians.csv"
        bundle_path = self.artifacts_dir / "pipeline_bundle.joblib"

        need_build = force_retrain or not (ranked_path.exists() and bundle_path.exists())
        if not need_build:
            chk = pd.read_parquet(ranked_path)
            if "region" not in chk.columns:
                need_build = True

        if need_build:
            run_ma_pipeline(
                n_companies=420,
                output_dir=self.artifacts_dir,
                use_live_yfinance=use_live,
                force_rebuild_data=force_retrain,
            )

        self.ranked_df = pd.read_parquet(ranked_path)
        if "data_source" not in self.ranked_df.columns:
            self.ranked_df["data_source"] = "Historical_Benchmark"

        self.comp_df = pd.read_csv(comp_path)
        self.lift_df = pd.read_csv(lift_path)
        self.rel_df = pd.read_csv(rel_path)
        self.sec_df = pd.read_csv(sec_path)
        self.bundle = joblib.load(bundle_path)
        self.last_updated_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


_SHARED_STATE: Optional[OrigonServiceState] = None


def get_shared_state(artifacts_dir: Path = ARTIFACTS_DIR) -> OrigonServiceState:
    global _SHARED_STATE
    if _SHARED_STATE is None or _SHARED_STATE.artifacts_dir != artifacts_dir:
        _SHARED_STATE = OrigonServiceState(artifacts_dir=artifacts_dir)
    return _SHARED_STATE


def create_app(artifacts_dir: Path = ARTIFACTS_DIR) -> Starlette:
    state = get_shared_state(artifacts_dir=artifacts_dir)

    def _localize_df(df: pd.DataFrame, lang: str) -> pd.DataFrame:
        out = df.copy()
        reg_map = get_region_map(lang)
        sec_map = get_sector_map(lang)
        out["clean_region"] = out["region"].map(lambda r: reg_map.get(str(r), sanitize_display_text(str(r))))
        out["clean_sector"] = out["sector"].map(lambda s: sec_map.get(str(s), sanitize_display_text(str(s))))
        out["clean_size"] = out["market_cap"].map(lambda m: assign_clean_size_bucket(float(m), lang=lang))
        return out

    async def get_health(request: Request) -> JSONResponse:
        lang = request.query_params.get("lang", "tr")
        return JSONResponse(
            {
                "status": "ok",
                "platform": "Origon Institutional M&A Analytics Platform"
                if is_en_lang(lang)
                else "Origon Kurumsal M&A Analitik Platformu",
                "champion_model": clean_model_title(state.bundle.get("champion_model_name", ""), lang=lang),
                "last_updated": state.last_updated_iso,
                "universe_count": int(len(state.ranked_df)),
            }
        )

    async def get_metadata(request: Request) -> JSONResponse:
        lang = request.query_params.get("lang", "tr")
        df = _localize_df(state.ranked_df, lang)
        return JSONResponse(
            {
                "regions": sorted(df["clean_region"].unique().tolist()),
                "sectors": sorted(df["clean_sector"].unique().tolist()),
                "size_buckets": get_size_buckets(lang),
                "max_probability_pct": round(float(df["predicted_ma_pct"].max()), 1),
                "min_leverage": round(float(df["net_debt_to_ebitda"].min()), 1),
                "max_leverage": round(float(df["net_debt_to_ebitda"].max()), 1),
            }
        )

    async def get_overview(request: Request) -> JSONResponse:
        lang = request.query_params.get("lang", "tr")
        df = _localize_df(state.ranked_df, lang)
        metrics = state.bundle["champion_metrics"]
        live_count = int((df["data_source"] == "Live_Yahoo_Finance").sum())
        high_conviction = int((df["predicted_ma_pct"] >= 35.0).sum())

        sector_summary = []
        for sec_name, grp in df.groupby("clean_sector"):
            sector_summary.append(
                {
                    "sector": str(sec_name),
                    "company_count": int(len(grp)),
                    "avg_probability_pct": round(float(grp["predicted_ma_pct"].mean()), 2),
                    "max_probability_pct": round(float(grp["predicted_ma_pct"].max()), 2),
                    "median_ev_ebitda": round(float(grp["ev_to_ebitda"].median()), 2),
                    "high_conviction_count": int((grp["predicted_ma_pct"] >= 35.0).sum()),
                }
            )
        sector_summary.sort(key=lambda x: x["avg_probability_pct"], reverse=True)

        region_summary = []
        for reg_name, grp in df.groupby("clean_region"):
            region_summary.append(
                {
                    "region": str(reg_name),
                    "company_count": int(len(grp)),
                    "avg_probability_pct": round(float(grp["predicted_ma_pct"].mean()), 2),
                    "median_ev_ebitda": round(float(grp["ev_to_ebitda"].median()), 2),
                }
            )

        return JSONResponse(
            {
                "total_universe": int(len(df)),
                "live_company_count": live_count,
                "high_conviction_count": high_conviction,
                "champion_model": clean_model_title(state.bundle.get("champion_model_name", ""), lang=lang),
                "last_updated": state.last_updated_iso,
                "metrics": {
                    "lift_top_10pct": round(float(metrics["lift_top_10pct"]), 2),
                    "precision_top_10pct": round(float(metrics["precision_top_10pct"]) * 100.0, 1),
                    "recall_top_10pct": round(float(metrics["recall_top_10pct"]) * 100.0, 1),
                    "base_rate_pct": round(float(metrics["base_rate"]) * 100.0, 2),
                    "pr_auc": round(float(metrics["pr_auc"]), 3),
                    "roc_auc": round(float(metrics["roc_auc"]), 3),
                    "brier_score": round(float(metrics["brier_score"]), 4),
                    "precision_at_10_pct": round(float(metrics["precision_at_10"]) * 100.0, 1),
                    "precision_at_25_pct": round(float(metrics["precision_at_25"]) * 100.0, 1),
                },
                "sector_summary": sector_summary,
                "region_summary": region_summary,
            }
        )

    async def get_targets(request: Request) -> JSONResponse:
        qp = request.query_params
        lang = qp.get("lang", "tr")
        df = _localize_df(state.ranked_df, lang)

        regions_param = qp.get("regions", "").strip()
        if regions_param:
            allowed_reg = {r.strip() for r in regions_param.split(",") if r.strip()}
            df = df[df["clean_region"].isin(allowed_reg)]

        sectors_param = qp.get("sectors", "").strip()
        if sectors_param:
            allowed_sec = {s.strip() for s in sectors_param.split(",") if s.strip()}
            df = df[df["clean_sector"].isin(allowed_sec)]

        sizes_param = qp.get("sizes", "").strip()
        if sizes_param:
            allowed_sz = {z.strip() for z in sizes_param.split(",") if z.strip()}
            df = df[df["clean_size"].isin(allowed_sz)]

        if qp.get("only_live", "false").lower() == "true":
            df = df[df["data_source"] == "Live_Yahoo_Finance"]

        min_prob = float(qp.get("min_prob", 0.0))
        max_lev = float(qp.get("max_leverage", 25.0))
        df = df[(df["predicted_ma_pct"] >= min_prob) & (df["net_debt_to_ebitda"] <= max_lev)]

        search_q = qp.get("q", "").strip().lower()
        if search_q:
            df = df[
                df["ticker"].str.lower().str.contains(search_q, na=False)
                | df["company_name"].str.lower().str.contains(search_q, na=False)
                | df["clean_sector"].str.lower().str.contains(search_q, na=False)
            ]

        sort_by = qp.get("sort_by", "rank")
        sort_dir = qp.get("sort_dir", "asc").lower()
        ascending = sort_dir == "asc"

        sort_col_map = {
            "rank": "ma_rank",
            "probability_pct": "predicted_ma_pct",
            "market_cap_usd_m": "market_cap",
            "ev_to_ebitda": "ev_to_ebitda",
            "sector_discount_pct": "sector_ev_ebitda_discount",
            "net_debt_to_ebitda": "net_debt_to_ebitda",
            "fcf_to_assets_pct": "fcf_to_assets",
        }
        target_col = sort_col_map.get(sort_by, "ma_rank")
        df = df.sort_values(by=target_col, ascending=ascending)

        total_filtered = int(len(df))
        limit = min(max(int(qp.get("limit", 200)), 1), 500)
        items = [serialize_target_row(row, lang=lang) for _, row in df.head(limit).iterrows()]

        return JSONResponse(
            {
                "total_filtered": total_filtered,
                "total_universe": int(len(state.ranked_df)),
                "items": items,
            }
        )

    async def get_target_detail(request: Request) -> JSONResponse:
        lang = request.query_params.get("lang", "tr")
        en = is_en_lang(lang)
        f_labels = get_feature_labels(lang)
        ticker = request.path_params["ticker"].upper().strip()
        df = state.ranked_df
        matches = df[df["ticker"].str.upper() == ticker]
        if matches.empty:
            msg = f"Company not found: {ticker}" if en else f"Şirket bulunamadı: {ticker}"
            return JSONResponse({"error": msg}, status_code=404)

        idx = int(matches.index[0])
        comp_row = df.loc[idx]
        comp_shap = state.bundle["sorted_shap_matrix"][idx]

        explainer = MAExplainer(model=state.bundle["champion_model"], feature_cols=ALL_RAW_FEATURES)
        exp_detail = explainer.explain_company(comp_row, comp_shap, lang=lang)

        shap_profile = []
        for feat_idx, feat_name in enumerate(ALL_RAW_FEATURES):
            s_val = float(comp_shap[feat_idx])
            shap_profile.append(
                {
                    "feature": feat_name,
                    "label": f_labels.get(feat_name, feat_name),
                    "shap_value": round(s_val, 4),
                    "abs_shap": round(abs(s_val), 4),
                    "direction": ("Positive" if s_val >= 0 else "Negative") if en else ("Pozitif" if s_val >= 0 else "Negatif"),
                }
            )
        shap_profile.sort(key=lambda x: x["abs_shap"], reverse=True)

        sec_df = state.sec_df
        if "region" in sec_df.columns:
            sub_sec = sec_df[
                (sec_df["region"] == comp_row["region"]) & (sec_df["sector"] == comp_row["sector"])
            ]
        else:
            sub_sec = sec_df[sec_df["sector"] == comp_row["sector"]]
        sec_med_row = sub_sec.iloc[0] if not sub_sec.empty else sec_df.iloc[0]

        base_info = serialize_target_row(comp_row, lang=lang)
        base_info.update(
            {
                "top_positive_drivers": exp_detail["top_positive_drivers"],
                "top_negative_deterrents": exp_detail["top_negative_deterrents"],
                "shap_profile": shap_profile[:14],
                "peer_benchmarks": build_peer_benchmarks(comp_row, sec_med_row, lang=lang),
            }
        )
        return JSONResponse(base_info)

    async def get_backtest(request: Request) -> JSONResponse:
        lang = request.query_params.get("lang", "tr")
        en = is_en_lang(lang)
        m_map = MODEL_DISPLAY_EN if en else MODEL_DISPLAY_TR
        s_map = STRATEGY_DISPLAY_EN if en else STRATEGY_DISPLAY_TR
        c_map = CALIBRATION_DISPLAY_EN if en else CALIBRATION_DISPLAY_TR

        models_list = []
        for _, r in state.comp_df.iterrows():
            raw_m = str(r["model_name"])
            raw_s = str(r["imbalance_strategy"])
            raw_c = str(r["calibration_method"])
            models_list.append(
                {
                    "model_name": m_map.get(raw_m, sanitize_display_text(raw_m)),
                    "imbalance_strategy": s_map.get(raw_s, raw_s),
                    "calibration_method": c_map.get(raw_c, raw_c),
                    "pr_auc": round(float(r["pr_auc"]), 4),
                    "roc_auc": round(float(r["roc_auc"]), 4),
                    "brier_score": round(float(r["brier_score"]), 4),
                    "lift_top_10pct": round(float(r["lift_top_10pct"]), 2),
                    "precision_top_10pct": round(float(r["precision_top_10pct"]) * 100.0, 1),
                    "recall_top_10pct": round(float(r["recall_top_10pct"]) * 100.0, 1),
                    "precision_at_10_pct": round(float(r["precision_at_10"]) * 100.0, 1),
                    "precision_at_25_pct": round(float(r["precision_at_25"]) * 100.0, 1),
                }
            )

        decile_list = state.lift_df.to_dict(orient="records")
        rel_list = []
        for _, r in state.rel_df.iterrows():
            raw_m = str(r["method"])
            rel_list.append(
                {
                    "method": c_map.get(raw_m, raw_m),
                    "method_key": raw_m,
                    "mean_predicted_prob": round(float(r["mean_predicted_prob"]), 4),
                    "observed_frequency": round(float(r["observed_frequency"]), 4),
                }
            )

        return JSONResponse(
            {
                "champion_model": clean_model_title(state.bundle.get("champion_model_name", ""), lang=lang),
                "models": models_list,
                "decile_lift": decile_list,
                "reliability_curve": rel_list,
            }
        )

    async def refresh_pipeline(request: Request) -> JSONResponse:
        use_live = True
        try:
            body = await request.json()
            use_live = bool(body.get("use_live_yfinance", True))
        except Exception:
            pass
        state.load_or_initialize(force_retrain=True, use_live=use_live)
        return JSONResponse(
            {
                "status": "refreshed",
                "last_updated": state.last_updated_iso,
                "total_universe": int(len(state.ranked_df)),
                "live_company_count": int((state.ranked_df["data_source"] == "Live_Yahoo_Finance").sum()),
            }
        )

    async def serve_index(request: Request):
        index_file = WEB_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"status": "Origon API running", "web_ui": "pending"}, status_code=200)

    routes: List[Any] = [
        Route("/api/v1/health", get_health, methods=["GET"]),
        Route("/api/v1/metadata", get_metadata, methods=["GET"]),
        Route("/api/v1/overview", get_overview, methods=["GET"]),
        Route("/api/v1/targets", get_targets, methods=["GET"]),
        Route("/api/v1/targets/{ticker}", get_target_detail, methods=["GET"]),
        Route("/api/v1/backtest", get_backtest, methods=["GET"]),
        Route("/api/v1/pipeline/refresh", refresh_pipeline, methods=["POST"]),
        Route("/", serve_index, methods=["GET"]),
    ]

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    routes.append(Mount("/static", app=StaticFiles(directory=str(WEB_DIR)), name="static"))

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    return Starlette(debug=False, routes=routes, middleware=middleware)


app = create_app()
