# Origon — Institutional M&A Decision Support and Origination Platform

Origon is an integrated machine learning, valuation, and stochastic deal simulation platform engineered for Investment Banking M&A Advisory, Corporate Development, and Private Equity teams. It unifies four core institutional workflows under a single high-density financial terminal with real-time bilingual (`TR` / `EN`) localization and support for both publicly listed corporations and user-entered private target companies.

---

## 1. Integrated Four-Module Architecture

### Module 1 — Multi-Region M&A Target Prediction and Causal SHAP Engine
- **Cross-Border FX and Regional Peer Normalization:** Converts balance sheets reported in `TRY`, `EUR`, `GBP`, `CHF`, `SEK`, and `NOK` into a unified **USD Millions** baseline via live Yahoo Finance exchange rates, while computing valuation discounts and robust Z-scores strictly within each `(region, sector, fiscal_year)` peer cell to neutralize Country Risk Premium bias.
- **22 Financial Indicators Across 5 Corporate Finance Hypotheses:** Undervaluation, Palepu Growth-Resource Mismatch, Leverage and Unused Debt Capacity, Profitability and Operational Synergy, and Scale/Market Dynamics.
- **Temporal Out-of-Time Calibration and TreeSHAP:** Trains Penalized ElasticNet Logistic Regression, XGBoost, and LightGBM on strictly time-split panels, calibrates annual probabilities via Platt Scaling and Isotonic Regression, and extracts the top three positive M&A drivers and top three negative deterrents per company.

### Module 2 — Strategic and Financial Buyer Matching Engine (`src/advisory/buyer_matcher.py`)
- **Acquirer Catalogs (`src/advisory/buyer_catalog.py`):**
  - **Strategic Buyers:** 32 global and regional large-cap corporate acquirers across all 8 sectors with cash reserves, EBITDA, leverage, and M&A expansion descriptors.
  - **Financial Sponsors:** 40 global and regional Private Equity funds grounded in public SEC Form ADV profiles, AUM, target equity ticket windows, focus sectors, and maximum leverage capacities.
- **Four-Dimension Matching Algorithm:**
  1. **NLP Vector Cosine Similarity (28%):** `TfidfVectorizer` bigram embeddings scoring horizontal and vertical industry value-chain fit.
  2. **Financial Capacity and Debt Headroom (30%):** Cash plus unused sector debt headroom relative to target deal enterprise value for strategic buyers; equity ticket window and leverage turn fit for PE sponsors.
  3. **Historical Deal Propensity and Scale Alignment (20%):** 5-year acquisition frequency, regional alignment, and deal-size preference fit.
  4. **Operational Synergy and Growth Complementarity (22%):** Mathematical complementarity between target revenue growth momentum and acquirer operating margin.

### Module 3 — Acquisition Premium and Offer Price Recommendation Engine (`src/advisory/premium_engine.py`)
- **Calibrated Precedent Transaction Dataset:** 640 historical M&A transactions calibrated to sectoral base premiums, valuation discounts, 52-week high drawdown anchors, 90-day equity volatility, growth rates, and interest rate regimes.
- **Quantile Gradient Boosting Regressors:** Predicts minimum ($\tau_{0.15}$), median ($\tau_{0.50}$), and maximum ($\tau_{0.85}$) control premiums, bridges standalone Market Cap to Implied Offer Equity Value and Implied Transaction Enterprise Value, models the target board's acceptance probability curve across offer premiums from $0\%$ to $75\%$, and lists comparable precedent deals.

### Module 4 — 5-Year Dynamic LBO and 5,000-Iteration Monte Carlo Risk Simulator (`src/advisory/lbo_simulator.py`)
- **Dynamic Capital Structure and Minimum Operating Cash Buffer Rule:** Models Senior and Subordinated debt tranches, interest rates, working capital, CapEx, and automated Free Cash Flow debt sweep while strictly enforcing a non-sweepable **Minimum Operating Cash Buffer** ($\text{MinCash}_t = \max(\alpha_{\text{buffer}} \times \text{Revenue}_t, \$10\text{M})$).
- **Vectorized Cholesky-Correlated Monte Carlo Engine:** Executes 5,000 correlated 5-year paths across revenue growth, EBITDA margin, floating interest rate, and exit multiple shocks in under 35 milliseconds, outputting the IRR and MoIC probability distributions, stress cash crunch probability, 5-year projection schedule, and a $5 \times 5$ Entry Leverage vs. Exit Multiple sensitivity matrix.

---

## 2. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health?lang=en\|tr` | Service health status, active champion model, and last update timestamp |
| `GET` | `/api/v1/metadata?lang=en\|tr` | Available market regions, sectors, scale buckets, and filter bounds |
| `GET` | `/api/v1/overview?lang=en\|tr` | Universe counts, live feed statistics, champion model KPIs, and sector summaries |
| `GET` | `/api/v1/targets?lang=en\|tr` | Filtered and sorted acquisition target list |
| `GET` | `/api/v1/targets/{ticker}?lang=en\|tr` | Company valuation metrics, top 3 positive/negative SHAP drivers, and regional peer benchmarks |
| `POST` | `/api/v1/advisory/buyers?lang=en\|tr` | Ranks top 5 overall, strategic, and PE sponsor buyers with 4-dimension sub-score breakdown |
| `POST` | `/api/v1/advisory/premium?lang=en\|tr` | Recommends control premium quantiles, offer price band, board acceptance curve, and precedent deals |
| `POST` | `/api/v1/advisory/lbo?lang=en\|tr` | Runs 5-year LBO schedule, 5,000-iteration Monte Carlo IRR distribution, and 5x5 sensitivity matrix |
| `GET` | `/api/v1/backtest?lang=en\|tr` | Model comparison table, 10-decile capture lift table, and calibration reliability curve |
| `POST` | `/api/v1/pipeline/refresh` | Triggers live market data refresh, model retraining, calibration, and SHAP computation |

---

## 3. Quick Start

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

On Windows, double-click `Origon.bat` to automatically start the server and open `http://127.0.0.1:8000` in your default browser.
