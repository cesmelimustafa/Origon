/**
 * Origon Institutional M&A Decision Support Platform - Unified 6-Module Client Application (TR / EN)
 */

import {
  fetchBacktest,
  fetchBuyerMatches,
  fetchLboSimulation,
  fetchMetadata,
  fetchOverview,
  fetchPremiumRecommendation,
  fetchTargetDetail,
  fetchTargets,
  triggerPipelineRefresh,
} from './api.js';

import {
  renderAcceptanceCurveChart,
  renderBuyerBreakdownChart,
  renderCalibrationCurveChart,
  renderDecileLiftChart,
  renderIrrHistogramChart,
  renderPremiumDriversChart,
  renderSectorProbabilityChart,
  renderShapDivergingChart,
  renderValuationScatterChart,
} from './charts.js';

const I18N = {
  tr: {
    brandTag: 'M&A ANALİTİK',
    brandSubtitle: 'Kurumsal Yatırım Bankacılığı Karar Destek Terminali',
    navScreener: 'Hedef Tarama',
    navDeepDive: 'Şirket Analizi',
    navBuyers: 'Alıcı Eşleştirme',
    navPremium: 'Prim ve Teklif Fiyatı',
    navLbo: 'LBO ve Risk Simülatörü',
    navBacktest: 'Model Doğrulama',
    btnRefresh: 'Piyasa Verisini Güncelle',
    btnRefreshing: 'Veri Güncelleniyor...',
    ctxBadgePublic: 'AKTİF HEDEF',
    ctxBadgeCustom: 'ÖZEL ŞİRKET',
    btnChainBuyers: 'Alıcı Eşleştirmeye Aktar',
    btnChainPremium: 'Teklif Fiyatlamasına Aktar',
    btnChainLbo: 'LBO Simülatörüne Aktar',
    btnCustomTarget: 'Özel Şirket Parametresi Gir',
    customDrawerTitle: 'Halka Açık Olmayan Özel Şirket Finansal Parametre Girişi',
    customDrawerMeta: 'Tüm değerler Milyon ABD Doları bazında işlenir ve üç modüle aynı anda aktarılır',
    lblCustName: 'Şirket Adı',
    lblCustRegion: 'Pazar Bölgesi',
    lblCustSector: 'Sektör',
    lblCustRev: 'Yıllık Ciro M$',
    lblCustEbitda: 'Yıllık FAVÖK M$',
    lblCustMcap: 'Baz Özsermaye Değeri M$',
    lblCustDebt: 'Toplam Borç M$',
    lblCustCash: 'Nakit ve Benzerleri M$',
    lblCustGrowth: 'Yıllık Ciro Büyümesi %',
    lblCustVol: '90 Günlük Volatilite %',
    lblCustDrawdown: 'Zirveye Uzaklık %',
    btnApplyCustom: 'Tüm Modüllerde Analiz Et',
    kpiUniverseLabel: 'Taranan Şirket Evreni',
    kpiLiveSuffix: 'Canlı Akış',
    kpiHighConvLabel: 'Yüksek Öncelikli Adaylar',
    kpiHighConvSub: 'Olasılık ≥ %35',
    kpiLiftLabel: 'İlk Yüzde On Yakalama Çarpanı',
    kpiBasePrefix: 'Baz Oran',
    kpiPrecisionLabel: 'İlk Yüzde On İsabet Oranı',
    kpiRecallPrefix: 'Yakalama',
    kpiAucLabel: 'Ayrıştırma Gücü PR ve ROC AUC',
    kpiAucSub: 'Dönem Dışı Test',
    kpiBrierLabel: 'Brier Olasılık Kalibrasyon Hatası',
    filterSearchTitle: 'Şirket veya Kod Arama',
    searchPlaceholder: 'Sembol, şirket veya sektör...',
    btnReset: 'Sıfırla',
    filterRegionTitle: 'Borsa ve Pazar Bölgesi',
    filterSectorTitle: 'Sektör Kırılımı',
    filterSizeTitle: 'Piyasa Değeri Ölçeği',
    filterMinProbTitle: 'Minimum Satın Alınma Olasılığı',
    filterMaxLevTitle: 'Maksimum Net Borç / FAVÖK',
    filterOnlyLiveTitle: 'Yalnızca Canlı Piyasa Akışı',
    tableMainTitle: 'Yıllık Satın Alınma Olasılığına Göre Sıralı Hedef Şirketler',
    tableCountSuffix: 'şirket listeleniyor · Seçmek için bir satıra tıklayın',
    thRank: 'Sıra',
    thTicker: 'Sembol',
    thCompany: 'Şirket Adı',
    thRegion: 'Pazar Bölgesi',
    thSector: 'Sektör',
    thSize: 'Ölçek',
    thProb: 'Satılma Olasılığı',
    thMcap: 'Piyasa Değeri M$',
    thEvEbitda: 'FD / FAVÖK',
    thDiscount: 'Sektör İskontosu',
    thLeverage: 'Net Borç / FAVÖK',
    thFcf: 'Serbest Nakit / Varlık',
    thRationale: 'Birincil Yatırım Gerekçesi',
    chartSectorTitle: 'Sektörlere Göre Ortalama Satın Alınma Olasılığı',
    chartSectorMeta: 'Bölgesel Normlu Dağılım',
    chartScatterTitle: 'FD / FAVÖK Çarpanı ve Satın Alınma Olasılığı İlişkisi',
    chartScatterMeta: 'Değerleme İskontosu Haritası',
    ddKpiProb: 'Yıllık Satın Alınma Olasılığı',
    ddRankPrefix: 'Sıralama',
    ddKpiMcap: 'Piyasa Değeri ve Firma Değeri',
    ddKpiEv: 'FD / FAVÖK ve Sektör İskontosu',
    ddDiscountPrefix: 'İskonto',
    ddKpiLev: 'Net Borç / FAVÖK',
    ddFcfPrefix: 'Serbest Nakit',
    ddKpiGrowth: 'Yıllık Ciro Büyümesi ve Marj',
    ddMarginPrefix: 'FAVÖK Marjı',
    ddKpiMarket: 'Pazar ve Raporlama Birimi',
    ddCurrencyPrefix: 'Bilanço Birimi',
    ddPosTitle: 'Satın Alınma Olasılığını Artıran Üç Temel Faktör',
    ddPosMeta: 'Pozitif SHAP Katkısı',
    ddNegTitle: 'Satın Alınma Olasılığını Sınırlayan Üç Temel Faktör',
    ddNegMeta: 'Negatif SHAP Etkisi',
    ddShapTitle: 'Finansal Faktörlerin SHAP Model Katkı Dağılımı',
    ddShapMeta: 'En Etkili On İki Değişken',
    ddPeerMeta: 'Emsal Sapma ve Standart Z-Skoru',
    thPeerMetric: 'Finansal Gösterge',
    thPeerCompany: 'Şirket Değeri',
    thPeerMedian: 'Bölgesel Medyan',
    thPeerDiff: 'Medyana Göre Fark',
    thPeerZ: 'Bölgesel Z-Skoru',
    bmKpiDealEv: 'Tahmini İşlem Firma Değeri',
    bmKpiDealSub: 'Kontrol Primi Dahil',
    bmKpiEquityTicket: 'Gerekli Fon Özsermaye Bileti',
    bmKpiEquitySub: 'LBO Sermaye Payı',
    bmKpiStratCount: 'Taranan Stratejik Alıcı',
    bmKpiStratSub: 'Kurumsal Bilanço Havuzu',
    bmKpiPeCount: 'Özel Sermaye Fonu Kataloğu',
    bmKpiPeSub: 'SEC Form ADV Profili',
    bmKpiTopBuyer: 'En Yüksek Uyumlu Alıcı',
    bmKpiTopScore: 'Lider Eşleşme Skoru',
    bmTableTitle: 'En Yüksek Uyum Skoruna Sahip İlk Beş Alıcı Adayı',
    bmFilterAll: 'Tüm Alıcılar',
    bmFilterStrat: 'Stratejik Alıcılar',
    bmFilterPe: 'Özel Sermaye Fonları',
    thBmRank: 'Sıra',
    thBmName: 'Alıcı Kurum veya Fon Adı',
    thBmType: 'Alıcı Tipi',
    thBmRegion: 'Bölge',
    thBmFirepower: 'Alım Kapasitesi M$',
    thBmScore: 'Toplam Uyum',
    thBmSubscores: 'Alt Boyut Kırılımı',
    bmChartTitle: 'İlk Beş Alıcı Adayının Dört Boyutlu Skor Kırılımı',
    bmChartMeta: 'Metin Benzerliği, Kapasite, Eğilim ve Sinerji',
    bmRationaleTitle: 'İlk Beş Alıcı Adayı İçin Stratejik ve Finansal Eşleşme Gerekçeleri',
    bmRationaleMeta: 'Kosinüs Vektör Uyumu ve Bilanço Borçlanma Kapasitesi Analizi',
    subNlp: 'Metin ve Sektör',
    subCap: 'Finansal Kapasite',
    subProp: 'İşlem Eğilimi',
    subSyn: 'Operasyonel Sinerji',
    prKpiBaseMcap: 'Mevcut Özsermaye ve Firma Değeri',
    prKpiMinBand: 'Minimum Teklif Primi ve Değeri',
    prKpiMedBand: 'Önerilen Medyan Teklif Primi',
    prKpiMaxBand: 'Maksimum Teklif Primi ve Değeri',
    prKpiTxMultiple: 'Zımni İşlem FD / FAVÖK Bandı',
    prKpiBoardProb: 'Medyan Teklifte Kabul Olasılığı',
    btnOfferToLbo: 'LBO Modülüne Aktar',
    prCurveTitle: 'Teklif Primi Yüzdesine Göre Yönetim Kurulu Kabul Olasılığı Eğrisi',
    prCurveMeta: 'Kantil Dağılımı ve 52 Haftalık Zirve Çıpası',
    prDriversTitle: 'Önerilen Kontrol Primini Belirleyen Değişkenlerin Etki Kırılımı',
    prDriversMeta: 'Sektör Baz Primine Eklenen Yüzde Puan Katkısı',
    prPrecedentTitle: 'Aynı Sektördeki Emsal Geçmiş M&A İşlemleri ve Çarpan Kıyaslaması',
    prPrecedentMeta: 'Tarihsel İşlem Veri Tablosu Eşleşmeleri',
    thPrDealId: 'İşlem Kodu',
    thPrYear: 'Yıl',
    thPrTarget: 'Hedef Şirket',
    thPrAcquirer: 'Alıcı Kurum veya Fon',
    thPrSector: 'Sektör',
    thPrDealEv: 'İşlem Büyüklüğü M$',
    thPrPreMult: 'İşlem Öncesi FD/FAVÖK',
    thPrTxMult: 'İşlem FD/FAVÖK',
    thPrPrem: 'Ödenen Kontrol Primi',
    lboCtrlEv: 'Giriş İşlem Değeri M$',
    lboCtrlLev: 'Toplam Borç / FAVÖK',
    lboCtrlSenShare: 'Kıdemli Kredi Payı',
    lboCtrlSenRate: 'Kıdemli Kredi Faizi',
    lboCtrlExitMult: 'Beşinci Yıl Çıkış Çarpanı',
    lboCtrlCashBuf: 'Min Nakit Tamponu Ciro %',
    lboKpiIrr: 'Beklenen ve Medyan IRR',
    lboKpiMoic: 'Yatırılan Sermaye Çarpanı MoIC',
    lboKpiHurdle: 'Eşik Getiri Yakalama Olasılığı',
    lboKpiHurdleSub: 'Hedef IRR ≥ %20',
    lboKpiDefault: 'Stres Nakit Sıkışması Olasılığı',
    lboKpiCapStruct: 'Sponsor Özsermaye ve Borç',
    lboKpiMinCash: 'Korunan Başlangıç Nakit Tamponu',
    lboKpiMinCashSub: 'Sıfırlanamaz Kasa Sınırı',
    lboHistTitle: '5.000 İterasyonlu Monte Carlo Yıllık İç Verim Oranı IRR Dağılımı',
    lboHistMeta: 'Cholesky Korelasyonlu Stokastik Simülasyon',
    lboSensTitle: 'Giriş Borçluluk Oranı ve Çıkış Çarpanı Getiri Duyarlılık Matrisi',
    lboSensMeta: 'Hücre Değerleri: Yıllık IRR Yüzdesi ve MoIC Çarpanı',
    lboSchedTitle: 'Beş Yıllık Dinamik LBO Nakit Akışı, Minimum Nakit Tamponu ve Otomatik Borç Azaltım Tablosu',
    lboSchedMeta: 'Kıdemli ve İkincil Kredi Dilimleri Nakit Süpürme Projeksiyonu',
    thLboYr: 'Projeksiyon Yılı',
    thLboRev: 'Ciro M$',
    thLboEbitda: 'FAVÖK M$',
    thLboInt: 'Nakit Faiz Gideri M$',
    thLboFcf: 'Serbest Nakit Akışı M$',
    thLboBuffer: 'Min Nakit Tamponu M$',
    thLboSweep: 'Otomatik Borç Ödemesi M$',
    thLboSenEnd: 'Kalan Kıdemli Borç M$',
    thLboSubEnd: 'Kalan İkincil Borç M$',
    thLboCashEnd: 'Dönem Sonu Nakit M$',
    thLboNetLev: 'Net Borç / FAVÖK',
    btTableTitle: 'Model Ailesi ve Sınıf Dengesizliği Stratejileri Karşılaştırması',
    btTableMeta: 'Zamana Dayalı Dönem Dışı Test Sonuçları',
    thBtModel: 'Model Mimarisi',
    thBtStrategy: 'Dengesizlik Yöntemi',
    thBtCalib: 'Olasılık Kalibrasyonu',
    thBtBrier: 'Brier Skoru',
    thBtLift10: 'İlk Yüzde On Lift',
    thBtPrec10Pct: 'İlk Yüzde On İsabet',
    thBtRec10Pct: 'İlk Yüzde On Yakalama',
    thBtP10: 'İlk On Şirket İsabeti',
    thBtP25: 'İlk Yirmi Beş İsabeti',
    btLiftChartTitle: 'Ondalık Dilimlere Göre M&A Yakalama Çarpanı',
    btLiftChartMeta: 'Birinci Dilim En Yüksek Olasılıklı Yüzde On',
    btCalChartTitle: 'Olasılık Kalibrasyon Güvenilirlik Eğrisi',
    btCalChartMeta: 'Tahmin Edilen Olasılık ve Gerçekleşen Frekans',
    btDecileTableTitle: 'Ondalık Dilim Bazında Kümülatif Yakalama Tablosu',
    btDecileTableMeta: 'Sıralama Performansı',
    thDecile: 'Olasılık Dilimi',
    thDecCompanies: 'Şirket Sayısı',
    thDecAcquired: 'Gerçekleşen M&A',
    thDecPred: 'Ortalama Tahmini Olasılık',
    thDecActual: 'Gerçekleşen M&A Oranı',
    thDecLift: 'Lift Çarpanı',
    thDecRecall: 'Kümülatif Yakalama Oranı',
    decilePrefix: 'Dilim',
    probOptionWord: 'Olasılık',
    sectorWord: 'Sektörü',
    peerHeaderJoin: 've',
    peerHeaderSuffix: 'Medyanı Kıyaslaması',
    yearPrefix: 'Yıl',
    sensCorner: 'Kaldıraç / Çıkış',
    lossProbPrefix: 'Anapara Kaybı',
    debtSharePrefix: 'Borç Oranı',
  },
  en: {
    brandTag: 'M&A ANALYTICS',
    brandSubtitle: 'Institutional Investment Banking Decision Support Terminal',
    navScreener: 'Target Screener',
    navDeepDive: 'Company Analysis',
    navBuyers: 'Buyer Matching',
    navPremium: 'Offer Valuation',
    navLbo: 'LBO Simulator',
    navBacktest: 'Model Validation',
    btnRefresh: 'Refresh Market Feed',
    btnRefreshing: 'Updating Feed...',
    ctxBadgePublic: 'ACTIVE TARGET',
    ctxBadgeCustom: 'PRIVATE TARGET',
    btnChainBuyers: 'Send to Buyer Match',
    btnChainPremium: 'Send to Offer Pricing',
    btnChainLbo: 'Send to LBO Simulator',
    btnCustomTarget: 'Custom Private Company',
    customDrawerTitle: 'Private or Unlisted Target Company Financial Parameter Entry',
    customDrawerMeta: 'All monetary figures are processed in USD Millions and propagated across all advisory modules',
    lblCustName: 'Company Name',
    lblCustRegion: 'Market Region',
    lblCustSector: 'Sector',
    lblCustRev: 'Annual Revenue $M',
    lblCustEbitda: 'Annual EBITDA $M',
    lblCustMcap: 'Base Equity Value $M',
    lblCustDebt: 'Total Debt $M',
    lblCustCash: 'Cash Balance $M',
    lblCustGrowth: 'Revenue Growth %',
    lblCustVol: '90-Day Volatility %',
    lblCustDrawdown: 'Drawdown from High %',
    btnApplyCustom: 'Evaluate Across All Modules',
    kpiUniverseLabel: 'Screened Corporate Universe',
    kpiLiveSuffix: 'Live Feed',
    kpiHighConvLabel: 'High Conviction Candidates',
    kpiHighConvSub: 'Probability ≥ 35%',
    kpiLiftLabel: 'Top Decile Capture Lift',
    kpiBasePrefix: 'Base Rate',
    kpiPrecisionLabel: 'Top Decile Precision',
    kpiRecallPrefix: 'Recall',
    kpiAucLabel: 'Discrimination Power PR and ROC AUC',
    kpiAucSub: 'Out-of-Time Test',
    kpiBrierLabel: 'Brier Probability Calibration Error',
    filterSearchTitle: 'Company or Ticker Search',
    searchPlaceholder: 'Ticker, company or sector...',
    btnReset: 'Reset',
    filterRegionTitle: 'Exchange and Market Region',
    filterSectorTitle: 'Sector Breakdown',
    filterSizeTitle: 'Market Cap Scale',
    filterMinProbTitle: 'Minimum Acquisition Probability',
    filterMaxLevTitle: 'Maximum Net Debt / EBITDA',
    filterOnlyLiveTitle: 'Live Market Feed Only',
    tableMainTitle: 'Target Companies Ranked by Annual Acquisition Probability',
    tableCountSuffix: 'companies listed · Click any row to select target',
    thRank: 'Rank',
    thTicker: 'Ticker',
    thCompany: 'Company Name',
    thRegion: 'Market Region',
    thSector: 'Sector',
    thSize: 'Scale',
    thProb: 'Acquisition Prob',
    thMcap: 'Market Cap $M',
    thEvEbitda: 'EV / EBITDA',
    thDiscount: 'Sector Discount',
    thLeverage: 'Net Debt / EBITDA',
    thFcf: 'FCF / Assets',
    thRationale: 'Primary Investment Rationale',
    chartSectorTitle: 'Mean Acquisition Probability by Sector',
    chartSectorMeta: 'Region-Normalized Distribution',
    chartScatterTitle: 'EV / EBITDA Multiple vs. Acquisition Probability',
    chartScatterMeta: 'Valuation Discount Map',
    ddKpiProb: 'Annual Acquisition Probability',
    ddRankPrefix: 'Rank',
    ddKpiMcap: 'Market Cap and Enterprise Value',
    ddKpiEv: 'EV / EBITDA and Sector Discount',
    ddDiscountPrefix: 'Discount',
    ddKpiLev: 'Net Debt / EBITDA',
    ddFcfPrefix: 'Free Cash Flow',
    ddKpiGrowth: 'Annual Revenue Growth and Margin',
    ddMarginPrefix: 'EBITDA Margin',
    ddKpiMarket: 'Market Region and Reporting Unit',
    ddCurrencyPrefix: 'Reporting Currency',
    ddPosTitle: 'Top Three Factors Increasing Acquisition Probability',
    ddPosMeta: 'Positive SHAP Contribution',
    ddNegTitle: 'Top Three Factors Constraining Acquisition Probability',
    ddNegMeta: 'Negative SHAP Impact',
    ddShapTitle: 'SHAP Model Contribution Profile Across Financial Factors',
    ddShapMeta: 'Top Twelve Drivers',
    ddPeerMeta: 'Peer Deviation and Robust Z-Score',
    thPeerMetric: 'Financial Indicator',
    thPeerCompany: 'Company Value',
    thPeerMedian: 'Regional Median',
    thPeerDiff: 'Spread vs Median',
    thPeerZ: 'Regional Z-Score',
    bmKpiDealEv: 'Implied Transaction EV',
    bmKpiDealSub: 'Including Control Premium',
    bmKpiEquityTicket: 'Required Sponsor Equity Ticket',
    bmKpiEquitySub: 'LBO Equity Contribution',
    bmKpiStratCount: 'Evaluated Strategic Buyers',
    bmKpiStratSub: 'Corporate Balance Sheet Pool',
    bmKpiPeCount: 'Private Equity Sponsor Catalog',
    bmKpiPeSub: 'SEC Form ADV Profiles',
    bmKpiTopBuyer: 'Highest Matching Acquirer',
    bmKpiTopScore: 'Lead Match Score',
    bmTableTitle: 'Top Five Ranked Acquirer Candidates by Total Fit Score',
    bmFilterAll: 'All Buyers',
    bmFilterStrat: 'Strategic Buyers',
    bmFilterPe: 'Private Equity Sponsors',
    thBmRank: 'Rank',
    thBmName: 'Acquirer or Sponsor Name',
    thBmType: 'Buyer Type',
    thBmRegion: 'Region',
    thBmFirepower: 'Firepower $M',
    thBmScore: 'Total Fit',
    thBmSubscores: 'Sub-Dimension Breakdown',
    bmChartTitle: 'Four-Dimension Fit Score Breakdown Across Top Five Acquirers',
    bmChartMeta: 'Text Vector Fit, Financial Capacity, Deal Propensity, and Synergy',
    bmRationaleTitle: 'Strategic and Financial Matching Rationales for Top Five Candidates',
    bmRationaleMeta: 'Cosine Vector Similarity and Balance Sheet Debt Headroom Analysis',
    subNlp: 'Text and Sector',
    subCap: 'Financial Capacity',
    subProp: 'Deal Propensity',
    subSyn: 'Operating Synergy',
    prKpiBaseMcap: 'Standalone Equity and Enterprise Value',
    prKpiMinBand: 'Minimum Offer Premium and Value',
    prKpiMedBand: 'Recommended Median Offer Premium',
    prKpiMaxBand: 'Maximum Offer Premium and Value',
    prKpiTxMultiple: 'Implied Transaction EV / EBITDA Band',
    prKpiBoardProb: 'Board Acceptance at Median Offer',
    btnOfferToLbo: 'Transfer to LBO',
    prCurveTitle: 'Target Board Acceptance Probability Curve Across Offer Premiums',
    prCurveMeta: 'Quantile Distribution and 52-Week High Anchor',
    prDriversTitle: 'Feature Attribution Breakdown Driving Recommended Control Premium',
    prDriversMeta: 'Percentage Point Contribution Above Sector Base Premium',
    prPrecedentTitle: 'Comparable Historical M&A Precedent Transactions in Target Sector',
    prPrecedentMeta: 'Calibrated Precedent Transaction Matches',
    thPrDealId: 'Deal ID',
    thPrYear: 'Year',
    thPrTarget: 'Target Company',
    thPrAcquirer: 'Acquirer or Sponsor',
    thPrSector: 'Sector',
    thPrDealEv: 'Deal Enterprise Value $M',
    thPrPreMult: 'Pre-Deal EV/EBITDA',
    thPrTxMult: 'Transaction EV/EBITDA',
    thPrPrem: 'Control Premium Paid',
    lboCtrlEv: 'Entry Transaction EV $M',
    lboCtrlLev: 'Total Debt / EBITDA',
    lboCtrlSenShare: 'Senior Debt Share',
    lboCtrlSenRate: 'Senior Debt Rate',
    lboCtrlExitMult: 'Year 5 Exit Multiple',
    lboCtrlCashBuf: 'Min Cash Buffer % Rev',
    lboKpiIrr: 'Expected and Median IRR',
    lboKpiMoic: 'Multiple on Invested Capital MoIC',
    lboKpiHurdle: 'Hurdle Rate Capture Probability',
    lboKpiHurdleSub: 'Target IRR ≥ 20%',
    lboKpiDefault: 'Stress Cash Crunch Probability',
    lboKpiCapStruct: 'Sponsor Equity and Total Debt',
    lboKpiMinCash: 'Protected Initial Cash Buffer',
    lboKpiMinCashSub: 'Non-Sweepable Cash Floor',
    lboHistTitle: '5,000-Iteration Monte Carlo Annual Internal Rate of Return IRR Distribution',
    lboHistMeta: 'Cholesky-Correlated Stochastic Simulation',
    lboSensTitle: 'Entry Leverage vs. Exit Multiple Return Sensitivity Matrix',
    lboSensMeta: 'Cell Values Display Annual IRR Percentage and MoIC Multiple',
    lboSchedTitle: 'Five-Year Dynamic LBO Cash Flow, Minimum Cash Buffer, and Debt Sweep Schedule',
    lboSchedMeta: 'Senior and Subordinated Debt Tranche Amortization Projection',
    thLboYr: 'Projection Year',
    thLboRev: 'Revenue $M',
    thLboEbitda: 'EBITDA $M',
    thLboInt: 'Cash Interest $M',
    thLboFcf: 'Free Cash Flow $M',
    thLboBuffer: 'Min Cash Buffer $M',
    thLboSweep: 'Debt Sweep Paydown $M',
    thLboSenEnd: 'Ending Senior Debt $M',
    thLboSubEnd: 'Ending Sub Debt $M',
    thLboCashEnd: 'Ending Cash $M',
    thLboNetLev: 'Net Debt / EBITDA',
    btTableTitle: 'Model Architecture and Class Imbalance Strategy Comparison',
    btTableMeta: 'Temporal Out-of-Time Validation Results',
    thBtModel: 'Model Architecture',
    thBtStrategy: 'Imbalance Strategy',
    thBtCalib: 'Probability Calibration',
    thBtBrier: 'Brier Score',
    thBtLift10: 'Top Decile Lift',
    thBtPrec10Pct: 'Top Decile Precision',
    thBtRec10Pct: 'Top Decile Recall',
    thBtP10: 'Precision at 10',
    thBtP25: 'Precision at 25',
    btLiftChartTitle: 'M&A Capture Lift Across Probability Deciles',
    btLiftChartMeta: 'Decile 1 Represents Highest-Scored Ten Percent',
    btCalChartTitle: 'Probability Calibration Reliability Curve',
    btCalChartMeta: 'Mean Predicted Probability vs. Observed Frequency',
    btDecileTableTitle: 'Cumulative Capture Performance by Probability Decile',
    btDecileTableMeta: 'Ranking Efficiency',
    thDecile: 'Probability Decile',
    thDecCompanies: 'Company Count',
    thDecAcquired: 'Realized M&A',
    thDecPred: 'Mean Predicted Prob',
    thDecActual: 'Realized M&A Rate',
    thDecLift: 'Lift Multiplier',
    thDecRecall: 'Cumulative Recall',
    decilePrefix: 'Decile',
    probOptionWord: 'Prob',
    sectorWord: 'Sector',
    peerHeaderJoin: 'vs.',
    peerHeaderSuffix: 'Median Benchmark',
    yearPrefix: 'Year',
    sensCorner: 'Leverage / Exit',
    lossProbPrefix: 'Capital Loss',
    debtSharePrefix: 'Debt Share',
  },
};

const state = {
  lang: 'tr',
  activeView: 'screener',
  metadata: null,
  overview: null,
  targets: [],
  selectedTicker: null,
  selectedDetail: null,
  isCustomMode: false,
  customTargetPayload: null,
  buyerMatches: null,
  buyerCategory: 'top_overall',
  premiumRec: null,
  lboResult: null,
  lboParams: {
    entry_ev_usd_m: 1500,
    total_debt_ebitda_turns: 4.5,
    senior_debt_share: 0.70,
    senior_interest_rate: 0.065,
    exit_ev_ebitda_multiple: 10.5,
    min_cash_buffer_pct_rev: 0.04,
  },
  backtest: null,
  filters: {
    regions: new Set(),
    sectors: new Set(),
    sizes: new Set(),
    onlyLive: false,
    minProb: 0,
    maxLeverage: 25,
    q: '',
    sortBy: 'rank',
    sortDir: 'asc',
  },
};

function t(key) {
  return (I18N[state.lang] && I18N[state.lang][key]) || I18N.tr[key] || key;
}

function getActiveTargetPayload() {
  if (state.isCustomMode && state.customTargetPayload) {
    return { custom_target: state.customTargetPayload };
  }
  return { ticker: state.selectedTicker || (state.targets[0] && state.targets[0].ticker) || 'MPARK.IS' };
}

function applyStaticTranslations() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    el.textContent = t(key);
  });
  const searchInput = document.getElementById('inputSearch');
  if (searchInput) searchInput.placeholder = t('searchPlaceholder');
}

function updateContextBar(summaryObj) {
  const badge = document.getElementById('ctxTargetBadge');
  badge.textContent = state.isCustomMode ? t('ctxBadgeCustom') : t('ctxBadgePublic');
  document.getElementById('ctxTargetTitle').textContent = `${summaryObj.ticker} — ${summaryObj.company_name}`;
  document.getElementById('ctxTargetRegion').textContent = summaryObj.region;
  document.getElementById('ctxTargetSector').textContent = summaryObj.sector;
  document.getElementById('ctxTargetEv').textContent = `EV: ${summaryObj.enterprise_value_usd_m.toFixed(0)}M$`;
  const ebitdaEst = summaryObj.enterprise_value_usd_m / Math.max(summaryObj.ev_to_ebitda, 1.0);
  document.getElementById('ctxTargetEbitda').textContent = `EBITDA: ${ebitdaEst.toFixed(0)}M$`;
}

function createChipGroup(containerId, items, activeSet, onChange) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  items.forEach((item) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `filter-chip ${activeSet.has(item) ? 'active' : ''}`;
    btn.textContent = item;
    btn.addEventListener('click', () => {
      if (activeSet.has(item)) {
        if (activeSet.size > 1) activeSet.delete(item);
      } else {
        activeSet.add(item);
      }
      createChipGroup(containerId, items, activeSet, onChange);
      onChange();
    });
    container.appendChild(btn);
  });
}

function updateKpiRibbon() {
  if (!state.overview) return;
  const ov = state.overview;
  const m = ov.metrics;

  document.getElementById('headerModelLabel').textContent = ov.champion_model;
  document.getElementById('kpiFilteredCount').textContent = `${state.targets.length} / ${ov.total_universe}`;
  document.getElementById('kpiLiveCount').textContent = `${ov.live_company_count} ${t('kpiLiveSuffix')}`;
  document.getElementById('kpiHighConviction').textContent = String(ov.high_conviction_count);
  document.getElementById('kpiLiftTop10').textContent = `${m.lift_top_10pct.toFixed(2)}x`;
  document.getElementById('kpiBaseRate').textContent = `${t('kpiBasePrefix')} %${m.base_rate_pct.toFixed(1)}`;
  document.getElementById('kpiPrecisionTop10').textContent = `%${m.precision_top_10pct.toFixed(1)}`;
  document.getElementById('kpiRecallTop10').textContent = `${t('kpiRecallPrefix')} %${m.recall_top_10pct.toFixed(1)}`;
  document.getElementById('kpiPrRocAuc').textContent = `${m.pr_auc.toFixed(3)} / ${m.roc_auc.toFixed(3)}`;
  document.getElementById('kpiBrierScore').textContent = m.brier_score.toFixed(4);
  document.getElementById('kpiUpdatedTime').textContent = ov.last_updated;
}

function renderTargetsTable() {
  const tbody = document.getElementById('targetsTableBody');
  tbody.innerHTML = '';

  document.getElementById('tableCountMeta').textContent = `${state.targets.length} ${t('tableCountSuffix')}`;
  const numLocale = state.lang === 'en' ? 'en-US' : 'tr-TR';

  state.targets.forEach((item) => {
    const tr = document.createElement('tr');
    if (!state.isCustomMode && item.ticker === state.selectedTicker) {
      tr.classList.add('selected');
    }

    const discClass = item.sector_discount_pct >= 0 ? 'val-pos' : 'val-neg';
    const discSign =
      item.sector_discount_pct >= 0
        ? `+${item.sector_discount_pct.toFixed(1)}`
        : item.sector_discount_pct.toFixed(1);
    const rankClass = item.rank <= 15 ? 'rank-badge top-tier' : 'rank-badge';
    const probWidth = Math.min(Math.max(item.probability_pct, 4), 100);

    tr.innerHTML = `
      <td class="num-col"><span class="${rankClass}">#${item.rank}</span></td>
      <td class="ticker-cell">${item.ticker}</td>
      <td>${item.company_name}</td>
      <td>${item.region}</td>
      <td>${item.sector}</td>
      <td>${item.size_bucket}</td>
      <td class="num-col">
        <div class="prob-meter">
          <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${probWidth}%"></div></div>
          <span>%${item.probability_pct.toFixed(2)}</span>
        </div>
      </td>
      <td class="num-col">${item.market_cap_usd_m.toLocaleString(numLocale, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}</td>
      <td class="num-col">${item.ev_to_ebitda.toFixed(2)}x</td>
      <td class="num-col ${discClass}">%${discSign}</td>
      <td class="num-col">${item.net_debt_to_ebitda.toFixed(2)}x</td>
      <td class="num-col">%${item.fcf_to_assets_pct.toFixed(1)}</td>
      <td>${item.primary_rationale}</td>
    `;

    tr.addEventListener('click', async () => {
      state.isCustomMode = false;
      state.selectedTicker = item.ticker;
      renderTargetsTable();
      await loadAllModulesForCurrentTarget(true);
    });

    tbody.appendChild(tr);
  });

  const sel = document.getElementById('selectTargetTicker');
  sel.innerHTML = '';
  state.targets.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.ticker;
    opt.textContent = `#${item.rank} · ${item.ticker} — ${item.company_name} · ${t('probOptionWord')} %${item.probability_pct.toFixed(2)}`;
    if (item.ticker === state.selectedTicker) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function loadSingleTargetDetail(ticker) {
  if (!ticker) return;
  const detail = await fetchTargetDetail(ticker, state.lang);
  state.selectedDetail = detail;
  state.selectedTicker = detail.ticker;

  if (!state.isCustomMode) {
    updateContextBar(detail);
  }

  document.getElementById('deepDiveTitle').textContent = `${detail.ticker} — ${detail.company_name}`;
  document.getElementById('deepDiveSubtitle').textContent = `${detail.region} · ${detail.sector} ${t('sectorWord')} · ${detail.data_source}`;

  document.getElementById('ddProb').textContent = `%${detail.probability_pct.toFixed(2)}`;
  document.getElementById('ddRank').textContent = `${t('ddRankPrefix')} #${detail.rank}`;
  document.getElementById('ddMcapEv').textContent = `${detail.market_cap_usd_m.toFixed(0)}M$ / ${detail.enterprise_value_usd_m.toFixed(0)}M$`;
  document.getElementById('ddSize').textContent = detail.size_bucket;
  document.getElementById('ddEvEbitda').textContent = `${detail.ev_to_ebitda.toFixed(2)}x`;
  const dSign =
    detail.sector_discount_pct >= 0
      ? `+${detail.sector_discount_pct.toFixed(1)}`
      : detail.sector_discount_pct.toFixed(1);
  document.getElementById('ddDiscount').textContent = `${t('ddDiscountPrefix')} %${dSign}`;
  document.getElementById('ddLeverage').textContent = `${detail.net_debt_to_ebitda.toFixed(2)}x`;
  document.getElementById('ddFcf').textContent = `${t('ddFcfPrefix')} %${detail.fcf_to_assets_pct.toFixed(1)}`;
  document.getElementById('ddGrowth').textContent = `%${detail.rev_growth_1y_pct.toFixed(1)}`;
  document.getElementById('ddMargin').textContent = `${t('ddMarginPrefix')} %${detail.ebitda_margin_pct.toFixed(1)}`;
  document.getElementById('ddRegion').textContent = detail.region;
  document.getElementById('ddCurrency').textContent = `${t('ddCurrencyPrefix')} ${detail.currency}`;

  const posContainer = document.getElementById('listPositiveDrivers');
  posContainer.innerHTML = '';
  detail.top_positive_drivers.forEach((drv, idx) => {
    const div = document.createElement('div');
    div.className = 'driver-item positive';
    div.innerHTML = `
      <div class="driver-top">
        <span class="driver-name">${idx + 1}. ${drv.label} — ${drv.formatted_value}</span>
        <span class="driver-shap pos">SHAP +${Math.abs(drv.shap_value).toFixed(3)}</span>
      </div>
      <div class="driver-desc">${drv.commentary}</div>
    `;
    posContainer.appendChild(div);
  });

  const negContainer = document.getElementById('listNegativeDeterrents');
  negContainer.innerHTML = '';
  detail.top_negative_deterrents.forEach((det, idx) => {
    const div = document.createElement('div');
    div.className = 'driver-item negative';
    div.innerHTML = `
      <div class="driver-top">
        <span class="driver-name">${idx + 1}. ${det.label} — ${det.formatted_value}</span>
        <span class="driver-shap neg">SHAP ${det.shap_value.toFixed(3)}</span>
      </div>
      <div class="driver-desc">${det.commentary}</div>
    `;
    negContainer.appendChild(div);
  });

  document.getElementById('peerTableHeader').textContent = `${detail.ticker} ${t('peerHeaderJoin')} ${detail.region} ${detail.sector} ${t('peerHeaderSuffix')}`;
  const peerBody = document.getElementById('peerBenchmarkBody');
  peerBody.innerHTML = '';
  detail.peer_benchmarks.forEach((row) => {
    const tr = document.createElement('tr');
    const diffClass = row.diff_pct >= 0 ? 'val-pos' : 'val-neg';
    tr.innerHTML = `
      <td>${row.label}</td>
      <td class="num-col">${row.company_formatted}</td>
      <td class="num-col">${row.median_formatted}</td>
      <td class="num-col ${diffClass}">${row.diff_formatted}</td>
      <td class="num-col">${row.z_formatted}</td>
    `;
    peerBody.appendChild(tr);
  });

  if (state.activeView === 'deepdive') {
    renderShapDivergingChart(document.getElementById('canvasShapBars'), detail.shap_profile);
  }
}

function renderBuyersView() {
  if (!state.buyerMatches) return;
  const bm = state.buyerMatches;
  const list = bm[state.buyerCategory] || bm.top_overall || [];

  document.getElementById('bmDealEv').textContent = `${bm.implied_deal_ev_usd_m.toLocaleString('tr-TR')}M$`;
  document.getElementById('bmEquityTicket').textContent = `${bm.required_sponsor_equity_usd_m.toLocaleString('tr-TR')}M$`;
  document.getElementById('bmStratPool').textContent = String(bm.strategic_pool_count);
  document.getElementById('bmPePool').textContent = String(bm.financial_pool_count);

  if (list.length > 0) {
    document.getElementById('bmTopBuyerName').textContent = list[0].name;
    document.getElementById('bmTopBuyerScore').textContent = `${list[0].total_score.toFixed(1)} / 100`;
    document.getElementById('bmTopBuyerType').textContent = list[0].buyer_type;
  }

  const tbody = document.getElementById('buyerMatchTableBody');
  tbody.innerHTML = '';
  const ratList = document.getElementById('buyerRationalesList');
  ratList.innerHTML = '';

  list.forEach((b, idx) => {
    const tr = document.createElement('tr');
    const bd = b.breakdown;
    tr.innerHTML = `
      <td class="num-col"><span class="rank-badge top-tier">#${idx + 1}</span></td>
      <td style="font-weight: 600;">${b.name}</td>
      <td>${b.buyer_type}</td>
      <td>${b.region}</td>
      <td class="num-col">${b.firepower_usd_m.toLocaleString('tr-TR', { maximumFractionDigits: 0 })}M$</td>
      <td class="num-col" style="color: var(--brand-gold-bright); font-weight: 700;">${b.total_score.toFixed(1)}</td>
      <td>
        <div class="subscore-stack">
          <div class="subscore-row">
            <span>${t('subNlp')}</span>
            <div class="subscore-track"><div class="subscore-fill" style="width:${bd.nlp_fit_score}%"></div></div>
            <span class="mono">${bd.nlp_fit_score.toFixed(0)}</span>
          </div>
          <div class="subscore-row">
            <span>${t('subCap')}</span>
            <div class="subscore-track"><div class="subscore-fill" style="width:${bd.financial_capacity_score}%"></div></div>
            <span class="mono">${bd.financial_capacity_score.toFixed(0)}</span>
          </div>
        </div>
      </td>
    `;
    tbody.appendChild(tr);

    const card = document.createElement('div');
    card.className = 'driver-item positive';
    card.innerHTML = `
      <div class="driver-top">
        <span class="driver-name">#${idx + 1} · ${b.name} — ${b.buyer_type} · ${b.focus_sector}</span>
        <span class="driver-shap pos">${b.total_score.toFixed(1)} / 100</span>
      </div>
      <div class="driver-desc">${b.rationale}</div>
    `;
    ratList.appendChild(card);
  });

  renderBuyerBreakdownChart(document.getElementById('canvasBuyerBreakdown'), list);
}

function renderPremiumView() {
  if (!state.premiumRec) return;
  const pr = state.premiumRec;

  document.getElementById('prStandaloneMcap').textContent = `${pr.standalone_market_cap_usd_m.toFixed(0)}M$`;
  document.getElementById('prStandaloneEv').textContent = `EV: ${pr.standalone_ev_usd_m.toFixed(0)}M$`;
  document.getElementById('prMinPrem').textContent = `+%${pr.min_premium_pct.toFixed(1)}`;
  document.getElementById('prMinEv').textContent = `EV: ${pr.offer_ev_min_usd_m.toFixed(0)}M$`;
  document.getElementById('prMedPrem').textContent = `+%${pr.median_premium_pct.toFixed(1)}`;
  document.getElementById('prMedEv').textContent = `EV: ${pr.offer_ev_median_usd_m.toFixed(0)}M$`;
  document.getElementById('prMaxPrem').textContent = `+%${pr.max_premium_pct.toFixed(1)}`;
  document.getElementById('prMaxEv').textContent = `EV: ${pr.offer_ev_max_usd_m.toFixed(0)}M$`;
  document.getElementById('prImpliedMultiples').textContent = `${pr.implied_tx_ev_ebitda_min.toFixed(1)}x – ${pr.implied_tx_ev_ebitda_max.toFixed(1)}x`;
  document.getElementById('prBaseMultiple').textContent = `Mevcut: ${pr.standalone_ev_ebitda.toFixed(1)}x`;
  document.getElementById('prBoardProb').textContent = `%${pr.board_acceptance_at_median_pct.toFixed(1)}`;

  const tbody = document.getElementById('precedentDealsBody');
  tbody.innerHTML = '';
  pr.precedent_deals.forEach((d) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="ticker-cell">${d.deal_id}</td>
      <td class="num-col">${d.deal_year}</td>
      <td>${d.target_name}</td>
      <td>${d.acquirer_name}</td>
      <td>${d.sector}</td>
      <td class="num-col">${d.deal_ev_usd_m.toFixed(1)}M$</td>
      <td class="num-col">${d.pre_deal_ev_ebitda.toFixed(2)}x</td>
      <td class="num-col">${d.transaction_ev_ebitda.toFixed(2)}x</td>
      <td class="num-col val-pos">+%${d.premium_paid_pct.toFixed(1)}</td>
    `;
    tbody.appendChild(tr);
  });

  renderAcceptanceCurveChart(document.getElementById('canvasAcceptanceCurve'), pr.acceptance_curve, pr.median_premium_pct);
  renderPremiumDriversChart(document.getElementById('canvasPremiumDrivers'), pr.premium_drivers);
}

function renderLboView() {
  if (!state.lboResult) return;
  const res = state.lboResult;
  const sum = res.summary;
  const cs = res.capital_structure;

  document.getElementById('lboMedIrr').textContent = `%${sum.median_irr_pct.toFixed(1)}`;
  document.getElementById('lboP10P90Irr').textContent = `P10 %${sum.p10_irr_pct.toFixed(0)} · P90 %${sum.p90_irr_pct.toFixed(0)}`;
  document.getElementById('lboMedMoic').textContent = `${sum.median_moic.toFixed(2)}x`;
  document.getElementById('lboP10P90Moic').textContent = `P10 ${sum.p10_moic.toFixed(2)}x · P90 ${sum.p90_moic.toFixed(2)}x`;
  document.getElementById('lboProbHurdle').textContent = `%${sum.prob_above_20pct_irr.toFixed(1)}`;
  document.getElementById('lboStressDefault').textContent = `%${sum.stress_default_probability_pct.toFixed(1)}`;
  document.getElementById('lboLossProb').textContent = `${t('lossProbPrefix')} %${sum.prob_capital_loss_pct.toFixed(1)}`;
  document.getElementById('lboEqDebtSplit').textContent = `${cs.sponsor_equity_usd_m.toFixed(0)}M$ / ${cs.total_debt_usd_m.toFixed(0)}M$`;
  document.getElementById('lboDebtSharePct').textContent = `${t('debtSharePrefix')} %${cs.debt_to_ev_pct.toFixed(1)}`;
  document.getElementById('lboMinCashVal').textContent = `${cs.initial_min_cash_buffer_usd_m.toFixed(1)}M$`;

  // 5-Year Schedule Table
  const schedBody = document.getElementById('lboScheduleBody');
  schedBody.innerHTML = '';
  res.projection_schedule.forEach((yr) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="num-col">${t('yearPrefix')} ${yr.year}</td>
      <td class="num-col">${yr.revenue_usd_m.toFixed(1)}</td>
      <td class="num-col">${yr.ebitda_usd_m.toFixed(1)}</td>
      <td class="num-col">${yr.interest_expense_usd_m.toFixed(1)}</td>
      <td class="num-col">${yr.free_cash_flow_usd_m.toFixed(1)}</td>
      <td class="num-col" style="color: var(--brand-gold-bright);">${yr.min_cash_buffer_usd_m.toFixed(1)}</td>
      <td class="num-col val-pos">${yr.debt_paydown_usd_m.toFixed(1)}</td>
      <td class="num-col">${yr.ending_senior_debt_usd_m.toFixed(1)}</td>
      <td class="num-col">${yr.ending_sub_debt_usd_m.toFixed(1)}</td>
      <td class="num-col">${yr.ending_cash_usd_m.toFixed(1)}</td>
      <td class="num-col">${yr.net_leverage_turns.toFixed(2)}x</td>
    `;
    schedBody.appendChild(tr);
  });

  // 5x5 Sensitivity Matrix Table
  const sensTable = document.getElementById('lboSensitivityTable');
  const mat = res.sensitivity_matrix;
  let html = `<thead><tr><th>${t('sensCorner')}</th>`;
  mat.exit_multiple_axes.forEach((mVal) => {
    html += `<th>${mVal.toFixed(1)}x</th>`;
  });
  html += `</tr></thead><tbody>`;

  mat.leverage_axes.forEach((lVal) => {
    html += `<tr><td style="font-weight:700; background:#121822;">${lVal.toFixed(2)}x</td>`;
    mat.exit_multiple_axes.forEach((mVal) => {
      const cell = mat.cells.find((c) => c.leverage_turns === lVal && c.exit_multiple === mVal);
      if (cell) {
        const cls =
          cell.irr_pct >= 20.0 ? 'sens-cell-strong' : cell.irr_pct >= 12.0 ? 'sens-cell-mid' : 'sens-cell-weak';
        html += `<td class="${cls}">%${cell.irr_pct.toFixed(1)} · ${cell.moic.toFixed(2)}x</td>`;
      } else {
        html += `<td>--</td>`;
      }
    });
    html += `</tr>`;
  });
  html += `</tbody>`;
  sensTable.innerHTML = html;

  renderIrrHistogramChart(document.getElementById('canvasIrrHistogram'), res.irr_histogram, res.summary);
}

async function runLboWithCurrentSliders() {
  const basePayload = getActiveTargetPayload();
  const lboPayload = {
    ...basePayload,
    ...state.lboParams,
  };
  state.lboResult = await fetchLboSimulation(lboPayload, state.lang);
  if (state.activeView === 'lbo') {
    renderLboView();
  }
}

async function loadAllModulesForCurrentTarget(syncLboEntryEv = true) {
  const payload = getActiveTargetPayload();

  if (!state.isCustomMode && state.selectedTicker) {
    await loadSingleTargetDetail(state.selectedTicker);
  }

  const [buyers, premium] = await Promise.all([
    fetchBuyerMatches(payload, state.lang),
    fetchPremiumRecommendation(payload, state.lang),
  ]);

  state.buyerMatches = buyers;
  state.premiumRec = premium;

  if (syncLboEntryEv && premium) {
    const recEv = Math.round(premium.offer_ev_median_usd_m);
    state.lboParams.entry_ev_usd_m = recEv;
    state.lboParams.exit_ev_ebitda_multiple = Math.min(Math.max(premium.implied_tx_ev_ebitda_median, 5.5), 19.0);

    const sliderEv = document.getElementById('sliderLboEv');
    sliderEv.max = String(Math.max(recEv * 2.5, 5000));
    sliderEv.value = String(recEv);
    document.getElementById('valLboEv').textContent = String(recEv);

    const sliderExit = document.getElementById('sliderLboExitMult');
    sliderExit.value = String(state.lboParams.exit_ev_ebitda_multiple.toFixed(1));
    document.getElementById('valLboExitMult').textContent = `${state.lboParams.exit_ev_ebitda_multiple.toFixed(1)}x`;
  }

  await runLboWithCurrentSliders();

  if (state.activeView === 'buyers') renderBuyersView();
  if (state.activeView === 'premium') renderPremiumView();
  if (state.activeView === 'lbo') renderLboView();
}

function renderBacktestView() {
  if (!state.backtest) return;
  const bt = state.backtest;

  const compBody = document.getElementById('modelComparisonBody');
  compBody.innerHTML = '';
  bt.models.forEach((m) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${m.model_name}</td>
      <td>${m.imbalance_strategy}</td>
      <td>${m.calibration_method}</td>
      <td class="num-col">${m.pr_auc.toFixed(4)}</td>
      <td class="num-col">${m.roc_auc.toFixed(4)}</td>
      <td class="num-col">${m.brier_score.toFixed(4)}</td>
      <td class="num-col">${m.lift_top_10pct.toFixed(2)}x</td>
      <td class="num-col">%${m.precision_top_10pct.toFixed(1)}</td>
      <td class="num-col">%${m.recall_top_10pct.toFixed(1)}</td>
      <td class="num-col">%${m.precision_at_10_pct.toFixed(1)}</td>
      <td class="num-col">%${m.precision_at_25_pct.toFixed(1)}</td>
    `;
    compBody.appendChild(tr);
  });

  const decBody = document.getElementById('decileTableBody');
  decBody.innerHTML = '';
  bt.decile_lift.forEach((d) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="num-col">${t('decilePrefix')} ${d.decile}</td>
      <td class="num-col">${d.n_companies}</td>
      <td class="num-col">${d.n_acquired}</td>
      <td class="num-col">%${(d.avg_predicted_prob * 100).toFixed(2)}</td>
      <td class="num-col">%${(d.actual_ma_rate * 100).toFixed(2)}</td>
      <td class="num-col">${d.lift.toFixed(2)}x</td>
      <td class="num-col">%${(d.cumulative_recall * 100).toFixed(1)}</td>
    `;
    decBody.appendChild(tr);
  });

  renderDecileLiftChart(document.getElementById('canvasDecileLift'), bt.decile_lift);
  renderCalibrationCurveChart(document.getElementById('canvasCalibrationCurve'), bt.reliability_curve);
}

function switchView(viewName) {
  state.activeView = viewName;
  document.querySelectorAll('.nav-tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === viewName);
  });
  document.querySelectorAll('.view-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `view-${viewName}`);
  });

  requestAnimationFrame(() => {
    if (viewName === 'screener') {
      if (state.overview) {
        renderSectorProbabilityChart(document.getElementById('canvasSectorProb'), state.overview.sector_summary);
      }
      renderValuationScatterChart(document.getElementById('canvasValuationScatter'), state.targets);
    } else if (viewName === 'deepdive' && state.selectedDetail) {
      renderShapDivergingChart(document.getElementById('canvasShapBars'), state.selectedDetail.shap_profile);
    } else if (viewName === 'buyers') {
      renderBuyersView();
    } else if (viewName === 'premium') {
      renderPremiumView();
    } else if (viewName === 'lbo') {
      renderLboView();
    } else if (viewName === 'backtest') {
      renderBacktestView();
    }
  });
}

async function applyFiltersAndReload() {
  const payload = await fetchTargets(
    {
      regions: Array.from(state.filters.regions),
      sectors: Array.from(state.filters.sectors),
      sizes: Array.from(state.filters.sizes),
      onlyLive: state.filters.onlyLive,
      minProb: state.filters.minProb,
      maxLeverage: state.filters.maxLeverage,
      q: state.filters.q,
      sortBy: state.filters.sortBy,
      sortDir: state.filters.sortDir,
    },
    state.lang
  );

  state.targets = payload.items;
  updateKpiRibbon();
  renderTargetsTable();

  if (state.activeView === 'screener') {
    if (state.overview) {
      renderSectorProbabilityChart(document.getElementById('canvasSectorProb'), state.overview.sector_summary);
    }
    renderValuationScatterChart(document.getElementById('canvasValuationScatter'), state.targets);
  }

  if (state.targets.length > 0 && !state.isCustomMode) {
    const exists = state.targets.some((item) => item.ticker === state.selectedTicker);
    const targetToLoad = exists ? state.selectedTicker : state.targets[0].ticker;
    state.selectedTicker = targetToLoad;
    await loadAllModulesForCurrentTarget(true);
  }
}

async function reloadForLanguage(newLang) {
  state.lang = newLang;
  document.querySelectorAll('.lang-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.lang === newLang);
  });

  applyStaticTranslations();

  const [meta, overview, backtest] = await Promise.all([
    fetchMetadata(state.lang),
    fetchOverview(state.lang),
    fetchBacktest(state.lang),
  ]);

  state.metadata = meta;
  state.overview = overview;
  state.backtest = backtest;

  state.filters.regions = new Set(meta.regions);
  state.filters.sectors = new Set(meta.sectors);
  state.filters.sizes = new Set(meta.size_buckets);

  createChipGroup('chipContainerRegions', meta.regions, state.filters.regions, applyFiltersAndReload);
  createChipGroup('chipContainerSectors', meta.sectors, state.filters.sectors, applyFiltersAndReload);
  createChipGroup('chipContainerSizes', meta.size_buckets, state.filters.sizes, applyFiltersAndReload);

  await applyFiltersAndReload();
  if (state.isCustomMode) {
    await loadAllModulesForCurrentTarget(false);
  }
  switchView(state.activeView);
}

async function initApp() {
  document.querySelectorAll('.lang-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetLang = btn.dataset.lang;
      if (targetLang !== state.lang) {
        reloadForLanguage(targetLang);
      }
    });
  });

  document.querySelectorAll('.nav-tab').forEach((btn) => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  // One-click chain navigation buttons in Active Target Context Bar
  document.getElementById('btnChainBuyers').addEventListener('click', () => switchView('buyers'));
  document.getElementById('btnChainPremium').addEventListener('click', () => switchView('premium'));
  document.getElementById('btnChainLbo').addEventListener('click', () => switchView('lbo'));

  document.getElementById('btnSendOfferToLbo').addEventListener('click', async () => {
    if (state.premiumRec) {
      const medEv = Math.round(state.premiumRec.offer_ev_median_usd_m);
      state.lboParams.entry_ev_usd_m = medEv;
      document.getElementById('sliderLboEv').value = String(medEv);
      document.getElementById('valLboEv').textContent = String(medEv);
      await runLboWithCurrentSliders();
    }
    switchView('lbo');
  });

  // Custom Private Target Drawer Toggle & Evaluation
  const customDrawer = document.getElementById('customTargetDrawer');
  document.getElementById('btnToggleCustomTarget').addEventListener('click', () => {
    customDrawer.classList.toggle('open');
  });

  document.getElementById('btnApplyCustomTarget').addEventListener('click', async () => {
    const rev = parseFloat(document.getElementById('custRev').value) || 500;
    const ebitda = parseFloat(document.getElementById('custEbitda').value) || 100;
    const mcap = parseFloat(document.getElementById('custMcap').value) || 750;
    const debt = parseFloat(document.getElementById('custDebt').value) || 180;
    const cash = parseFloat(document.getElementById('custCash').value) || 60;
    const growthPct = parseFloat(document.getElementById('custGrowth').value) || 15;
    const volPct = parseFloat(document.getElementById('custVol').value) || 28;
    const ddPct = parseFloat(document.getElementById('custDrawdown').value) || -18;
    const compName = document.getElementById('custName').value.trim() || 'Özel Hedef Şirket';
    const regVal = document.getElementById('custRegion').value;
    const secVal = document.getElementById('custSector').value;

    state.isCustomMode = true;
    state.customTargetPayload = {
      ticker: 'PRIVATE',
      company_name: compName,
      region: regVal,
      sector: secVal,
      revenue: rev,
      ebitda: ebitda,
      market_cap: mcap,
      total_debt: debt,
      cash_and_equivalents: cash,
      rev_growth_1y: growthPct / 100.0,
      stock_volatility_90d: volPct / 100.0,
      dist_from_52w_high: ddPct / 100.0,
    };

    const ev = Math.max(mcap + debt - cash, ebitda * 3);
    updateContextBar({
      ticker: 'PRIVATE',
      company_name: compName,
      region: document.getElementById('custRegion').selectedOptions[0].text,
      sector: document.getElementById('custSector').selectedOptions[0].text,
      enterprise_value_usd_m: ev,
      ev_to_ebitda: ev / Math.max(ebitda, 1),
    });

    await loadAllModulesForCurrentTarget(true);
    customDrawer.classList.remove('open');
    if (state.activeView === 'screener' || state.activeView === 'deepdive') {
      switchView('buyers');
    }
  });

  // Buyer category filter chips
  document.querySelectorAll('[data-buyer-cat]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.buyerCategory = btn.getAttribute('data-buyer-cat');
      document.querySelectorAll('[data-buyer-cat]').forEach((b) => {
        b.classList.toggle('active', b === btn);
      });
      renderBuyersView();
    });
  });

  // LBO Interactive Sliders
  const bindLboSlider = (sliderId, valId, paramKey, formatter, parser = parseFloat) => {
    const el = document.getElementById(sliderId);
    el.addEventListener('input', () => {
      const v = parser(el.value);
      state.lboParams[paramKey] = v;
      document.getElementById(valId).textContent = formatter(v);
    });
    el.addEventListener('change', () => {
      runLboWithCurrentSliders();
    });
  };

  bindLboSlider('sliderLboEv', 'valLboEv', 'entry_ev_usd_m', (v) => `${v.toFixed(0)}`);
  bindLboSlider('sliderLboLev', 'valLboLev', 'total_debt_ebitda_turns', (v) => `${v.toFixed(2)}x`);
  bindLboSlider(
    'sliderLboSenShare',
    'valLboSenShare',
    'senior_debt_share',
    (v) => `%${(v * 100).toFixed(0)}`,
    (raw) => parseFloat(raw) / 100.0
  );
  bindLboSlider(
    'sliderLboSenRate',
    'valLboSenRate',
    'senior_interest_rate',
    (v) => `%${(v * 100).toFixed(2)}`,
    (raw) => parseFloat(raw) / 100.0
  );
  bindLboSlider('sliderLboExitMult', 'valLboExitMult', 'exit_ev_ebitda_multiple', (v) => `${v.toFixed(1)}x`);
  bindLboSlider(
    'sliderLboCashBuf',
    'valLboCashBuf',
    'min_cash_buffer_pct_rev',
    (v) => `%${(v * 100).toFixed(1)}`,
    (raw) => parseFloat(raw) / 100.0
  );

  // Screener search and sliders
  const inputSearch = document.getElementById('inputSearch');
  inputSearch.addEventListener('input', () => {
    state.filters.q = inputSearch.value.trim();
    applyFiltersAndReload();
  });

  const sliderMinProb = document.getElementById('sliderMinProb');
  sliderMinProb.addEventListener('input', () => {
    state.filters.minProb = parseFloat(sliderMinProb.value);
    document.getElementById('valMinProb').textContent = `%${state.filters.minProb.toFixed(1)}`;
    applyFiltersAndReload();
  });

  const sliderMaxLev = document.getElementById('sliderMaxLeverage');
  sliderMaxLev.addEventListener('input', () => {
    state.filters.maxLeverage = parseFloat(sliderMaxLev.value);
    document.getElementById('valMaxLeverage').textContent = `${state.filters.maxLeverage.toFixed(1)}x`;
    applyFiltersAndReload();
  });

  const chkOnlyLive = document.getElementById('chkOnlyLive');
  chkOnlyLive.addEventListener('change', () => {
    state.filters.onlyLive = chkOnlyLive.checked;
    applyFiltersAndReload();
  });

  document.getElementById('btnResetFilters').addEventListener('click', () => {
    if (!state.metadata) return;
    state.filters.regions = new Set(state.metadata.regions);
    state.filters.sectors = new Set(state.metadata.sectors);
    state.filters.sizes = new Set(state.metadata.size_buckets);
    state.filters.onlyLive = false;
    state.filters.minProb = 0;
    state.filters.maxLeverage = 25;
    state.filters.q = '';
    inputSearch.value = '';
    sliderMinProb.value = '0';
    sliderMaxLev.value = '25';
    chkOnlyLive.checked = false;
    document.getElementById('valMinProb').textContent = '%0.0';
    document.getElementById('valMaxLeverage').textContent = '25.0x';
    createChipGroup('chipContainerRegions', state.metadata.regions, state.filters.regions, applyFiltersAndReload);
    createChipGroup('chipContainerSectors', state.metadata.sectors, state.filters.sectors, applyFiltersAndReload);
    createChipGroup('chipContainerSizes', state.metadata.size_buckets, state.filters.sizes, applyFiltersAndReload);
    applyFiltersAndReload();
  });

  document.querySelectorAll('#targetsTable thead th.sortable').forEach((th) => {
    th.addEventListener('click', () => {
      const col = th.dataset.sort;
      if (state.filters.sortBy === col) {
        state.filters.sortDir = state.filters.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        state.filters.sortBy = col;
        state.filters.sortDir = col === 'rank' ? 'asc' : 'desc';
      }
      applyFiltersAndReload();
    });
  });

  document.getElementById('selectTargetTicker').addEventListener('change', async (e) => {
    state.isCustomMode = false;
    state.selectedTicker = e.target.value;
    await loadAllModulesForCurrentTarget(true);
  });

  const btnRefresh = document.getElementById('btnRefreshPipeline');
  btnRefresh.addEventListener('click', async () => {
    const label = document.getElementById('btnRefreshText');
    label.textContent = t('btnRefreshing');
    btnRefresh.disabled = true;
    try {
      await triggerPipelineRefresh();
      await reloadForLanguage(state.lang);
    } finally {
      label.textContent = t('btnRefresh');
      btnRefresh.disabled = false;
    }
  });

  window.addEventListener('resize', () => {
    switchView(state.activeView);
  });

  await reloadForLanguage('tr');
}

document.addEventListener('DOMContentLoaded', initApp);
