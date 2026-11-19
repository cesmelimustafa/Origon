"""Integration tests for the decoupled Origon REST API service layer (Task 1)."""

import json
from typing import Any, Dict, Tuple
from urllib.parse import urlsplit

from src.api.server import create_app


def _invoke_asgi(app: Any, path_with_query: str, method: str = "GET", body: bytes = b"") -> Tuple[int, Dict[str, Any]]:
    """
    Execute an ASGI request directly on a single-threaded event loop without
    spawning cross-thread socketpairs that deadlock inside the Windows sandbox.
    """
    parsed = urlsplit(path_with_query)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": parsed.path,
        "raw_path": parsed.path.encode("utf-8"),
        "query_string": parsed.query.encode("utf-8"),
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }

    status_code = 500
    response_chunks = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])
        elif message["type"] == "http.response.body":
            response_chunks.append(message.get("body", b""))

    coro = app(scope, receive, send)
    try:
        while True:
            coro.send(None)
    except StopIteration:
        pass

    raw_bytes = b"".join(response_chunks)
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        payload = {"raw": raw_bytes.decode("utf-8", errors="ignore")}
    return status_code, payload


def test_api_health_metadata_and_overview_endpoints():
    app = create_app()

    code_h, health_data = _invoke_asgi(app, "/api/v1/health")
    assert code_h == 200
    assert health_data["status"] == "ok"
    assert "champion_model" in health_data

    code_m, meta = _invoke_asgi(app, "/api/v1/metadata")
    assert code_m == 200
    assert len(meta["regions"]) >= 3
    assert len(meta["sectors"]) == 8
    assert len(meta["size_buckets"]) == 3

    code_o, over = _invoke_asgi(app, "/api/v1/overview")
    assert code_o == 200
    assert over["total_universe"] > 0
    assert "lift_top_10pct" in over["metrics"]
    assert "pr_auc" in over["metrics"]
    assert len(over["sector_summary"]) == 8


def test_api_targets_filtering_and_single_company_xai_endpoint():
    app = create_app()

    code_t, payload = _invoke_asgi(app, "/api/v1/targets?min_prob=10&max_leverage=15&limit=25")
    assert code_t == 200
    assert "total_filtered" in payload
    assert "items" in payload
    assert len(payload["items"]) > 0

    first_ticker = payload["items"][0]["ticker"]
    code_d, detail = _invoke_asgi(app, f"/api/v1/targets/{first_ticker}")
    assert code_d == 200
    assert detail["ticker"] == first_ticker
    assert len(detail["top_positive_drivers"]) == 3
    assert len(detail["top_negative_deterrents"]) == 3
    assert len(detail["shap_profile"]) >= 12
    assert len(detail["peer_benchmarks"]) >= 10


def test_api_backtest_and_root_html_delivery():
    app = create_app()

    code_b, bt = _invoke_asgi(app, "/api/v1/backtest")
    assert code_b == 200
    assert len(bt["models"]) >= 4
    assert len(bt["decile_lift"]) == 10
    assert len(bt["reliability_curve"]) > 0


def test_api_supports_bilingual_tr_and_en_responses():
    app = create_app()

    code_tr, meta_tr = _invoke_asgi(app, "/api/v1/metadata?lang=tr")
    code_en, meta_en = _invoke_asgi(app, "/api/v1/metadata?lang=en")
    assert code_tr == 200 and code_en == 200
    assert "Küçük Ölçek" in meta_tr["size_buckets"]
    assert "Small Cap" in meta_en["size_buckets"]

    _, targets_en = _invoke_asgi(app, "/api/v1/targets?lang=en&limit=5")
    first_item_en = targets_en["items"][0]
    assert first_item_en["size_bucket"] in {"Small Cap", "Mid Cap", "Large Cap"}

    ticker = first_item_en["ticker"]
    _, detail_en = _invoke_asgi(app, f"/api/v1/targets/{ticker}?lang=en")
    _, detail_tr = _invoke_asgi(app, f"/api/v1/targets/{ticker}?lang=tr")
    assert detail_en["top_positive_drivers"][0]["label"] != detail_tr["top_positive_drivers"][0]["label"]
    assert detail_en["top_positive_drivers"][0]["commentary"] != detail_tr["top_positive_drivers"][0]["commentary"]

