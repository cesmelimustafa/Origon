/**
 * Origon Institutional M&A Analytics Platform - Bilingual Client Application (TR / EN)
 */

import {
  fetchBacktest,
  fetchMetadata,
  fetchOverview,
  fetchTargetDetail,
  fetchTargets,
  triggerPipelineRefresh,
} from './api.js';

import {
  renderCalibrationCurveChart,
  renderDecileLiftChart,
  renderSectorProbabilityChart,
  renderShapDivergingChart,
  renderValuationScatterChart,
} from './charts.js';

const I18N = {
  tr: {
    brandTag: 'M&A ANALİTİK',
    brandSubtitle: 'Kurumsal Hedef Şirket Tahmin ve Karar Destek Terminali',
    navScreener: 'Hedef Tarama Terminali',
    navDeepDive: 'Şirket Analizi ve Karar Gerekçesi',
    navBacktest: 'Model Doğrulama ve Kalibrasyon',
    btnRefresh: 'Piyasa Verisini Güncelle',
    btnRefreshing: 'Veri Güncelleniyor...',
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
    tableCountSuffix: 'şirket listeleniyor · Detaylı analiz için bir satıra tıklayın',
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
  },
  en: {
    brandTag: 'M&A ANALYTICS',
    brandSubtitle: 'Institutional Acquisition Target Prediction and Decision Terminal',
    navScreener: 'Target Screening Terminal',
    navDeepDive: 'Company Analysis and Deal Rationale',
    navBacktest: 'Model Validation and Calibration',
    btnRefresh: 'Refresh Market Feed',
    btnRefreshing: 'Updating Feed...',
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
    tableCountSuffix: 'companies listed · Click any row for detailed causal analysis',
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

function applyStaticTranslations() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    el.textContent = t(key);
  });
  const searchInput = document.getElementById('inputSearch');
  if (searchInput) searchInput.placeholder = t('searchPlaceholder');
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
    if (item.ticker === state.selectedTicker) {
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
      state.selectedTicker = item.ticker;
      await loadSingleTargetDetail(item.ticker);
      switchView('deepdive');
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

  renderShapDivergingChart(document.getElementById('canvasShapBars'), detail.shap_profile);
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

  if (state.targets.length > 0) {
    const exists = state.targets.some((item) => item.ticker === state.selectedTicker);
    const targetToLoad = exists ? state.selectedTicker : state.targets[0].ticker;
    state.selectedTicker = targetToLoad;
    await loadSingleTargetDetail(targetToLoad);
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
  if (state.activeView === 'backtest') {
    renderBacktestView();
  }
}

async function initApp() {
  // Language toggle buttons
  document.querySelectorAll('.lang-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetLang = btn.dataset.lang;
      if (targetLang !== state.lang) {
        reloadForLanguage(targetLang);
      }
    });
  });

  // Navigation tabs
  document.querySelectorAll('.nav-tab').forEach((btn) => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  // Search input
  const inputSearch = document.getElementById('inputSearch');
  inputSearch.addEventListener('input', () => {
    state.filters.q = inputSearch.value.trim();
    applyFiltersAndReload();
  });

  // Sliders
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

  // Reset button
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

  // Table sorting
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

  // Deep-dive target select
  document.getElementById('selectTargetTicker').addEventListener('change', async (e) => {
    await loadSingleTargetDetail(e.target.value);
  });

  // Live refresh button
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
