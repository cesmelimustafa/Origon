# Origon — Institutional M&A Target Prediction and Deal Origination Platform

Origon is an end-to-end machine learning and causal explainability platform designed for Investment Banking M&A Advisory and Corporate Development teams. It analyzes financial statements and valuation multiples of publicly listed corporations across the United States, Europe, and Turkey Borsa Istanbul to predict twelve-month acquisition target probabilities and generate company-level deal rationales.

---

## 1. Core Architectural Pillars

### Multi-Market Coverage and Cross-Border Normalization
- **Live FX Normalization (`src/data/fetcher.py`):** Financial statements and market capitalizations reported in `TRY`, `EUR`, `GBP`, `CHF`, `SEK`, and `NOK` are converted into a unified **USD Millions** baseline using live foreign exchange rates from Yahoo Finance. This eliminates currency scale distortion in asset-size features.
- **Regional Sector Normalization (`src/features/financial_ratios.py`, `src/features/sector_normalizer.py`):** To prevent Country Risk Premium and interest-rate regime bias across markets, valuation discounts and robust Z-scores are computed strictly within each `(region, sector, fiscal_year)` peer cell.

### Five Corporate Finance Hypothesis Categories (22 Engineered Features)
1. **Undervaluation Hypothesis:** `ev_to_ebitda`, `price_to_book`, `price_to_sales`, `sector_ev_ebitda_discount`, `sector_pb_discount`.
2. **Growth-Resource Mismatch (Palepu Framework):** `rev_growth_1y`, `rev_growth_3y_cagr`, `fcf_to_assets`, `cash_to_assets`, `capex_to_revenue`, `growth_resource_mismatch`.
3. **Leverage and Unused Debt Capacity:** `net_debt_to_ebitda`, `debt_to_equity`, `current_ratio`, `interest_coverage`.
4. **Profitability and Operational Synergy:** `ebitda_margin`, `roa`, `roe`, `asset_turnover`, `margin_momentum_4q`.
5. **Scale and Market Dynamics:** `log_total_assets`, `stock_return_12m`, `stock_volatility_12m`, `sector_hhi`.

### Leakage-Free Temporal Splitting, Calibration, and SHAP XAI
- **Temporal Train / Calibration / Out-of-Time Test Split (`TimeBasedSplitter`):** Training (`fiscal_year <= 2022`), Probability Calibration (`fiscal_year == 2023`), and Out-of-Time Evaluation (`fiscal_year >= 2024`). Winsorization bounds (`1%–99%`) and sector medians are fit strictly on the training fold.
- **Model Family and Imbalance Handling:** Penalized ElasticNet Logistic Regression, XGBoost, and LightGBM evaluated under cost-sensitive `scale_pos_weight` and training-fold `SMOTE`.
- **Post-Hoc Probability Calibration (`src/models/calibration.py`):** Platt Sigmoid Scaling and Isotonic Regression calibrated on the validation period to produce accurate annual acquisition probabilities evaluated via Brier Score, PR-AUC, ROC-AUC, Precision@K, and Top-Decile Capture Lift (`Lift@Top10%`).
- **Causal Explainability (`src/models/explainer.py`):** Extracts the top three positive M&A drivers and top three negative deterrents per company using TreeSHAP and LinearSHAP, generating bilingual (`TR` / `EN`) institutional deal rationales.

---

## 2. Project Directory Structure

```text
ma-target-prediction/
├── .gitignore                        # Git exclusion rules
├── Origon.bat                        # One-click Windows launcher for API and web terminal
├── README.md                         # Project documentation
├── requirements.txt                  # Pinned Python dependencies
├── run_tests.py                      # Test suite runner
├── artifacts/                        # Trained models, calibrators, SHAP matrices, and panel cache
├── design-system/
│   └── origon/
│       └── MASTER.md                 # Origon institutional terminal design system specification
├── src/
│   ├── config.py                     # Multi-region universe, feature definitions, and split parameters
│   ├── pipeline.py                   # End-to-end training, calibration, evaluation, and SHAP orchestrator
│   ├── api/
│   │   ├── schemas.py                # Bilingual (TR / EN) institutional serializer
│   │   └── server.py                 # Decoupled ASGI REST API service and static web host
│   ├── data/
│   │   ├── benchmark_generator.py    # Multi-region historical M&A panel generator
│   │   ├── fetcher.py                # Live Yahoo Finance fetcher and FX normalizer
│   │   └── preprocessor.py           # Temporal splitter and leakage-free winsorization
│   ├── features/
│   │   ├── financial_ratios.py       # Vectorized corporate finance ratio engine
│   │   └── sector_normalizer.py      # Cross-sectional regional peer robust Z-score transformer
│   └── models/
│       ├── calibration.py            # Platt Scaling and Isotonic Regression probability calibration
│       ├── evaluator.py              # PR-AUC, ROC-AUC, Brier Score, Precision@K, and Decile Lift
│       ├── explainer.py              # Bilingual SHAP deal rationale engine
│       ├── imbalance.py              # Class weighting and SMOTE handlers
│       └── trainer.py                # Logistic Regression, XGBoost, and LightGBM trainers
├── tests/                            # Unit and REST API integration test suite
└── web/                              # Origon Institutional Financial Web Terminal (TR / EN)
    ├── index.html
    ├── assets/
    ├── css/
    │   └── terminal.css
    └── js/
        ├── api.js
        ├── app.js
        └── charts.js
```

---

## 3. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health?lang=en\|tr` | Service health status, active champion model, and last update timestamp |
| `GET` | `/api/v1/metadata?lang=en\|tr` | Available market regions, sectors, scale buckets, and filter bounds |
| `GET` | `/api/v1/overview?lang=en\|tr` | Universe counts, live feed statistics, champion model KPIs, and sector summaries |
| `GET` | `/api/v1/targets?lang=en\|tr` | Filtered and sorted acquisition target list |
| `GET` | `/api/v1/targets/{ticker}?lang=en\|tr` | Company valuation metrics, top 3 positive/negative SHAP drivers, and regional peer benchmarks |
| `GET` | `/api/v1/backtest?lang=en\|tr` | Model comparison table, 10-decile capture lift table, and calibration reliability curve |
| `POST` | `/api/v1/pipeline/refresh` | Triggers live market data refresh, model retraining, calibration, and SHAP computation |

---

## 4. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end training, calibration, and SHAP pipeline
python -m src.pipeline

# 3. Run the unit and API integration test suite
python run_tests.py

# 4. Launch the Origon REST API and Web Terminal on http://127.0.0.1:8000
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

On Windows, double-click `Origon.bat` to automatically start the server and open `http://127.0.0.1:8000` in your default browser. Use the `TR | EN` language switcher in the top header bar to toggle the interface between Turkish and English in real time.
