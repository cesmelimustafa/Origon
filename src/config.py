"""Global configuration and constants for the M&A Target Prediction System."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

REGIONS: List[str] = [
    "US (NYSE/NASDAQ)",
    "Europe (Euronext/XETRA/LSE)",
    "Turkey (Borsa İstanbul - BIST)",
]

SECTORS: List[str] = [
    "Technology",
    "Healthcare",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Materials",
]

MARKET_TICKERS_BY_REGION_AND_SECTOR: Dict[str, Dict[str, List[str]]] = {
    "US (NYSE/NASDAQ)": {
        "Technology": ["AAPL", "MSFT", "ADBE", "CRM", "INTC", "AMD", "ORCL", "CSCO", "NOW", "SNPS"],
        "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "BMY", "GILD", "AMGN", "BIIB", "INCY", "EXEL"],
        "Industrials": ["HON", "CAT", "GE", "DE", "EMR", "ETN", "ROK", "PH", "DOV", "XYL"],
        "Consumer Discretionary": ["AMZN", "HD", "NKE", "MCD", "SBUX", "TGT", "EBAY", "ETSY", "HAS", "WHR"],
        "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "GIS", "K", "CPB", "SJM", "HRL"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "DVN", "FANG", "APA", "CTRA", "OVV"],
        "Financials": ["JPM", "BAC", "GS", "MS", "BLK", "SCHW", "TROW", "IVZ", "BEN", "AMG"],
        "Materials": ["LIN", "APD", "SHW", "NEM", "FCX", "DD", "DOW", "EMN", "CE", "ALB"],
    },
    "Europe (Euronext/XETRA/LSE)": {
        "Technology": ["SAP.DE", "ASML.AS", "CAP.PA", "IFX.DE", "STMPA.PA", "NOKIA.HE", "LOGN.SW", "DSY.PA"],
        "Healthcare": ["NOVN.SW", "ROG.SW", "SAN.PA", "AZN.L", "GSK.L", "BAYN.DE", "MRK.DE", "ipn.pa"],
        "Industrials": ["SIE.DE", "AIR.PA", "DPW.DE", "SU.PA", "ALO.PA", "VOLV-B.ST", "SAND.ST", "RHM.DE"],
        "Consumer Discretionary": ["MC.PA", "RMS.PA", "ADS.DE", "PUM.DE", "BMW.DE", "MBG.DE", "VOW3.DE", "INDITEX.MC"],
        "Consumer Staples": ["NESN.SW", "ULVR.L", "OR.PA", "BN.PA", "ABI.BR", "HEIA.AS", "DGE.L", "TSCO.L"],
        "Energy": ["SHEL.L", "TTE.PA", "BP.L", "EQNR.OL", "ENI.MI", "REP.MC", "OMV.VI", "GALP.LS"],
        "Financials": ["BNP.PA", "SAN.MC", "DBK.DE", "INGA.AS", "ACA.PA", "ALV.DE", "AXA.PA", "BARC.L"],
        "Materials": ["BAS.DE", "AI.PA", "LIN.DE", "AKZA.AS", "RIO.L", "AAL.L", "GLEN.L", "CRH.L"],
    },
    "Turkey (Borsa İstanbul - BIST)": {
        "Technology": ["ASELS.IS", "LOGO.IS", "MIATK.IS", "ARDYZ.IS", "INDES.IS", "KFEIN.IS", "SMART.IS", "VBTYZ.IS"],
        "Healthcare": ["MPARK.IS", "SELEC.IS", "ECILC.IS", "GENIL.IS", "DEVA.IS", "LKMNH.IS", "MEDTR.IS", "TNZTP.IS"],
        "Industrials": ["THYAO.IS", "SISE.IS", "FROTO.IS", "TOASO.IS", "TAVHL.IS", "TKFEN.IS", "ENKAI.IS", "OTKAR.IS"],
        "Consumer Discretionary": ["ARCLK.IS", "VESTL.IS", "DOAS.IS", "MAVI.IS", "BJKAS.IS", "FENER.IS", "YATAS.IS", "VAKKO.IS"],
        "Consumer Staples": ["BIMAS.IS", "MGROS.IS", "SOKM.IS", "ULKER.IS", "CCOLA.IS", "AEFES.IS", "TATGD.IS", "PETUN.IS"],
        "Energy": ["TUPRS.IS", "ENJSA.IS", "AKSEN.IS", "ZOREN.IS", "AYGAZ.IS", "ODAS.IS", "GWIND.IS", "SMRTG.IS"],
        "Financials": ["GARAN.IS", "AKBNK.IS", "ISCTR.IS", "YKBNK.IS", "KCHOL.IS", "SAHOL.IS", "VAKBN.IS", "HALKB.IS"],
        "Materials": ["EREGL.IS", "KRDMD.IS", "PETKM.IS", "SASA.IS", "CIMSA.IS", "AKCNS.IS", "OYAKC.IS", "GUBRF.IS"],
    },
}

DEFAULT_TICKERS_BY_SECTOR: Dict[str, List[str]] = MARKET_TICKERS_BY_REGION_AND_SECTOR["US (NYSE/NASDAQ)"]

DEFAULT_FX_TO_USD: Dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "GBp": 0.0127,  # LSE pence
    "CHF": 1.13,
    "SEK": 0.095,
    "NOK": 0.093,
    "DKK": 0.145,
    "TRY": 0.0285,
}

FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "Undervaluation": [
        "ev_to_ebitda",
        "price_to_book",
        "price_to_sales",
        "sector_ev_ebitda_discount",
        "sector_pb_discount",
    ],
    "Growth_Resource_Mismatch": [
        "rev_growth_1y",
        "rev_growth_3y_cagr",
        "fcf_to_assets",
        "cash_to_assets",
        "capex_to_revenue",
        "growth_resource_mismatch",
    ],
    "Leverage_Liquidity": [
        "net_debt_to_ebitda",
        "debt_to_equity",
        "current_ratio",
        "interest_coverage",
    ],
    "Profitability_Efficiency": [
        "ebitda_margin",
        "roa",
        "roe",
        "asset_turnover",
        "margin_momentum_4q",
    ],
    "Scale_Market_Dynamics": [
        "log_total_assets",
        "stock_return_12m",
        "stock_volatility_12m",
        "sector_hhi",
    ],
}

ALL_RAW_FEATURES: List[str] = [
    feat for group in FEATURE_CATEGORIES.values() for feat in group
]

FEATURE_LABELS_TR: Dict[str, str] = {
    "ev_to_ebitda": "FD / FAVÖK",
    "price_to_book": "Piyasa Değeri / Defter Değeri",
    "price_to_sales": "Piyasa Değeri / Satışlar",
    "sector_ev_ebitda_discount": "Bölgesel Sektör FD/FAVÖK İskontosu",
    "sector_pb_discount": "Bölgesel Sektör PD/DD İskontosu",
    "rev_growth_1y": "Yıllık Ciro Büyümesi",
    "rev_growth_3y_cagr": "Üç Yıllık Bileşik Ciro Büyümesi",
    "fcf_to_assets": "Serbest Nakit Akışı / Varlıklar",
    "cash_to_assets": "Nakit / Toplam Varlıklar",
    "capex_to_revenue": "Sermaye Harcaması / Ciro",
    "growth_resource_mismatch": "Büyüme ve Kaynak Uyumsuzluk Endeksi",
    "net_debt_to_ebitda": "Net Borç / FAVÖK",
    "debt_to_equity": "Toplam Borç / Özsermaye",
    "current_ratio": "Cari Oran",
    "interest_coverage": "Faiz Karşılama Oranı",
    "ebitda_margin": "FAVÖK Marjı",
    "roa": "Aktif Karlılığı",
    "roe": "Özsermaye Karlılığı",
    "asset_turnover": "Varlık Devir Hızı",
    "margin_momentum_4q": "Dört Çeyrek FAVÖK Marj İvmesi",
    "log_total_assets": "Firma Ölçeği",
    "stock_return_12m": "Yıllık Hisse Getirisi",
    "stock_volatility_12m": "Yıllık Hisse Volatilitesi",
    "sector_hhi": "Sektörel Yoğunlaşma Endeksi",
}

FEATURE_LABELS_EN: Dict[str, str] = {
    "ev_to_ebitda": "EV / EBITDA",
    "price_to_book": "Price / Book Value",
    "price_to_sales": "Price / Sales",
    "sector_ev_ebitda_discount": "Regional Sector EV/EBITDA Discount",
    "sector_pb_discount": "Regional Sector P/B Discount",
    "rev_growth_1y": "Annual Revenue Growth",
    "rev_growth_3y_cagr": "Three-Year Revenue CAGR",
    "fcf_to_assets": "Free Cash Flow / Total Assets",
    "cash_to_assets": "Cash / Total Assets",
    "capex_to_revenue": "Capital Expenditure / Revenue",
    "growth_resource_mismatch": "Growth and Resource Mismatch Index",
    "net_debt_to_ebitda": "Net Debt / EBITDA",
    "debt_to_equity": "Total Debt / Equity",
    "current_ratio": "Current Ratio",
    "interest_coverage": "Interest Coverage Ratio",
    "ebitda_margin": "EBITDA Margin",
    "roa": "Return on Assets",
    "roe": "Return on Equity",
    "asset_turnover": "Asset Turnover",
    "margin_momentum_4q": "Four-Quarter EBITDA Margin Momentum",
    "log_total_assets": "Firm Asset Scale",
    "stock_return_12m": "Twelve-Month Equity Return",
    "stock_volatility_12m": "Twelve-Month Equity Volatility",
    "sector_hhi": "Regional Sector Concentration Index",
}


@dataclass(frozen=True)
class SplitConfig:
    """Temporal train / calibration / out-of-time test configuration."""

    train_end_year: int = 2022
    val_year: int = 2023
    test_start_year: int = 2024
    winsor_limits: Tuple[float, float] = (0.01, 0.99)
    target_col: str = "is_acquired"
    year_col: str = "fiscal_year"
    sector_col: str = "sector"
    ticker_col: str = "ticker"


DEFAULT_SPLIT_CONFIG = SplitConfig()
