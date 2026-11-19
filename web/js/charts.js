/**
 * Origon Institutional High-DPI Canvas Chart Engine
 * Renders Screener, SHAP, Buyer Breakdown, Board Acceptance Curve, Premium Attribution, and Monte Carlo IRR Histograms.
 */

const COLORS = {
  bg: '#0F141C',
  grid: '#1F2937',
  text: '#F1F5F9',
  muted: '#94A3B8',
  gold: '#D4AF37',
  goldBright: '#EBC85B',
  cobalt: '#3B82F6',
  positive: '#10B981',
  negative: '#F43F5E',
};

function setupHiDPICanvas(canvas) {
  if (!canvas || !canvas.parentElement) return null;
  const rect = canvas.parentElement.getBoundingClientRect();
  const width = Math.max(rect.width - 20, 280);
  const height = Math.max(rect.height - 20, 220);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

export function renderSectorProbabilityChart(canvas, sectorSummary = []) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !sectorSummary.length) return;
  const { ctx, width, height } = setup;

  const padLeft = 155;
  const padRight = 56;
  const padTop = 16;
  const padBottom = 26;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const maxVal = Math.max(...sectorSummary.map((d) => d.avg_probability_pct), 25);
  const rowH = plotH / sectorSummary.length;

  ctx.font = '500 11px "JetBrains Mono", monospace';
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;

  for (let i = 0; i <= 4; i++) {
    const x = padLeft + (plotW * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, height - padBottom);
    ctx.stroke();

    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = 'center';
    ctx.fillText(`%${((maxVal * i) / 4).toFixed(0)}`, x, height - 8);
  }

  sectorSummary.forEach((item, idx) => {
    const yCenter = padTop + idx * rowH + rowH * 0.5;
    const barH = Math.min(rowH * 0.56, 18);
    const barW = Math.max((item.avg_probability_pct / maxVal) * plotW, 4);

    ctx.fillStyle = COLORS.text;
    ctx.font = '500 11.5px "Inter", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(item.sector, padLeft - 12, yCenter);

    const grad = ctx.createLinearGradient(padLeft, 0, padLeft + barW, 0);
    grad.addColorStop(0, COLORS.cobalt);
    grad.addColorStop(1, COLORS.gold);
    ctx.fillStyle = grad;
    ctx.fillRect(padLeft, yCenter - barH / 2, barW, barH);

    ctx.fillStyle = COLORS.goldBright;
    ctx.font = '600 11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`%${item.avg_probability_pct.toFixed(1)}`, padLeft + barW + 8, yCenter);
  });
}

export function renderValuationScatterChart(canvas, items = []) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !items.length) return;
  const { ctx, width, height } = setup;

  const padLeft = 52;
  const padRight = 24;
  const padTop = 18;
  const padBottom = 34;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const sample = items.slice(0, 160);
  const maxX = Math.min(Math.max(...sample.map((d) => d.ev_to_ebitda), 20), 40);
  const maxY = Math.max(...sample.map((d) => d.probability_pct), 50);

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.font = '500 10.5px "JetBrains Mono", monospace';

  for (let i = 0; i <= 4; i++) {
    const y = padTop + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);
    ctx.stroke();

    const valY = maxY * (1 - i / 4);
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`%${valY.toFixed(0)}`, padLeft - 8, y);
  }

  for (let i = 0; i <= 4; i++) {
    const x = padLeft + (plotW * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, height - padBottom);
    ctx.stroke();

    const valX = (maxX * i) / 4;
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = 'center';
    ctx.fillText(`${valX.toFixed(0)}x`, x, height - 12);
  }

  sample.forEach((d) => {
    const clampedEv = Math.min(Math.max(d.ev_to_ebitda, 0), maxX);
    const x = padLeft + (clampedEv / maxX) * plotW;
    const y = padTop + plotH - (Math.min(d.probability_pct, maxY) / maxY) * plotH;
    const radius = Math.min(Math.max(Math.log10(Math.max(d.market_cap_usd_m, 50)) * 1.6, 3), 7);

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    if (d.region.includes('Türkiye') || d.region.includes('Turkey')) {
      ctx.fillStyle = 'rgba(212, 175, 55, 0.82)';
    } else if (d.region.includes('Avrupa') || d.region.includes('European')) {
      ctx.fillStyle = 'rgba(16, 185, 129, 0.78)';
    } else {
      ctx.fillStyle = 'rgba(59, 130, 246, 0.78)';
    }
    ctx.fill();
  });
}

export function renderShapDivergingChart(canvas, shapProfile = []) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !shapProfile.length) return;
  const { ctx, width, height } = setup;

  const rows = shapProfile.slice(0, 12);
  const padLeft = 248;
  const padRight = 58;
  const padTop = 14;
  const padBottom = 24;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.shap_value)), 0.25);
  const zeroX = padLeft + plotW * 0.5;
  const rowH = plotH / rows.length;

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(zeroX, padTop);
  ctx.lineTo(zeroX, height - padBottom);
  ctx.stroke();

  rows.forEach((item, idx) => {
    const yCenter = padTop + idx * rowH + rowH * 0.5;
    const barH = Math.min(rowH * 0.58, 17);
    const halfW = plotW * 0.48;
    const w = (Math.abs(item.shap_value) / maxAbs) * halfW;

    ctx.fillStyle = COLORS.text;
    ctx.font = '500 11.5px "Inter", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(item.label, padLeft - 12, yCenter);

    const isPos = item.shap_value >= 0;
    ctx.fillStyle = isPos ? COLORS.positive : COLORS.negative;
    const barX = isPos ? zeroX : zeroX - w;
    ctx.fillRect(barX, yCenter - barH / 2, Math.max(w, 2), barH);

    ctx.font = '600 11px "JetBrains Mono", monospace';
    ctx.textAlign = isPos ? 'left' : 'right';
    const labelX = isPos ? zeroX + w + 6 : zeroX - w - 6;
    const signStr = item.shap_value >= 0 ? `+${item.shap_value.toFixed(3)}` : item.shap_value.toFixed(3);
    ctx.fillText(signStr, labelX, yCenter);
  });
}

export function renderBuyerBreakdownChart(canvas, buyers = []) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !buyers.length) return;
  const { ctx, width, height } = setup;

  const top5 = buyers.slice(0, 5);
  const padLeft = 185;
  const padRight = 50;
  const padTop = 28;
  const padBottom = 26;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const rowH = plotH / top5.length;

  const subColors = [COLORS.cobalt, COLORS.gold, COLORS.positive, '#8B5CF6'];
  const subKeys = ['nlp_fit_score', 'financial_capacity_score', 'deal_propensity_score', 'synergy_score'];
  const weights = [0.28, 0.30, 0.20, 0.22];

  top5.forEach((b, idx) => {
    const yCenter = padTop + idx * rowH + rowH * 0.5;
    const barH = Math.min(rowH * 0.48, 22);

    ctx.fillStyle = COLORS.text;
    ctx.font = '500 11.5px "Inter", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const shortName = b.name.length > 25 ? `${b.name.slice(0, 24)}…` : b.name;
    ctx.fillText(shortName, padLeft - 12, yCenter);

    let curX = padLeft;
    subKeys.forEach((k, sIdx) => {
      const weightedPts = (b.breakdown[k] || 0) * weights[sIdx];
      const segW = (weightedPts / 100.0) * plotW;
      ctx.fillStyle = subColors[sIdx];
      ctx.fillRect(curX, yCenter - barH / 2, Math.max(segW - 1, 1), barH);
      curX += segW;
    });

    ctx.fillStyle = COLORS.goldBright;
    ctx.font = '600 11.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${b.total_score.toFixed(1)}`, curX + 8, yCenter);
  });
}

export function renderAcceptanceCurveChart(canvas, curve = [], medianPrem = 30.0) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !curve.length) return;
  const { ctx, width, height } = setup;

  const padLeft = 52;
  const padRight = 24;
  const padTop = 20;
  const padBottom = 32;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const maxPrem = 75.0;

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.font = '500 10.5px "JetBrains Mono", monospace';

  for (let i = 0; i <= 4; i++) {
    const y = padTop + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);
    ctx.stroke();

    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`%${(100 - i * 25).toFixed(0)}`, padLeft - 8, y);
  }

  for (let p = 0; p <= 75; p += 15) {
    const x = padLeft + (p / maxPrem) * plotW;
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = 'center';
    ctx.fillText(`+%${p}`, x, height - 10);
  }

  // Median recommended offer vertical marker
  const medX = padLeft + (Math.min(medianPrem, maxPrem) / maxPrem) * plotW;
  ctx.strokeStyle = COLORS.gold;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(medX, padTop);
  ctx.lineTo(medX, padTop + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Sigmoid curve path
  ctx.strokeStyle = COLORS.positive;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  curve.forEach((pt, idx) => {
    const x = padLeft + (pt.premium_pct / maxPrem) * plotW;
    const y = padTop + plotH - (pt.acceptance_probability_pct / 100.0) * plotH;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

export function renderPremiumDriversChart(canvas, drivers = []) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !drivers.length) return;
  const { ctx, width, height } = setup;

  const padLeft = 225;
  const padRight = 56;
  const padTop = 16;
  const padBottom = 24;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.impact_pct_pts)), 4.0);
  const zeroX = padLeft + plotW * 0.5;
  const rowH = plotH / drivers.length;

  ctx.strokeStyle = COLORS.grid;
  ctx.beginPath();
  ctx.moveTo(zeroX, padTop);
  ctx.lineTo(zeroX, height - padBottom);
  ctx.stroke();

  drivers.forEach((d, idx) => {
    const yCenter = padTop + idx * rowH + rowH * 0.5;
    const barH = Math.min(rowH * 0.56, 18);
    const halfW = plotW * 0.46;
    const w = (Math.abs(d.impact_pct_pts) / maxAbs) * halfW;

    ctx.fillStyle = COLORS.text;
    ctx.font = '500 11.5px "Inter", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(d.label, padLeft - 12, yCenter);

    const isPos = d.impact_pct_pts >= 0;
    ctx.fillStyle = isPos ? COLORS.positive : COLORS.negative;
    const barX = isPos ? zeroX : zeroX - w;
    ctx.fillRect(barX, yCenter - barH / 2, Math.max(w, 2), barH);

    ctx.font = '600 11px "JetBrains Mono", monospace';
    ctx.textAlign = isPos ? 'left' : 'right';
    const lx = isPos ? zeroX + w + 6 : zeroX - w - 6;
    const signStr = isPos ? `+%${d.impact_pct_pts.toFixed(1)}` : `%${d.impact_pct_pts.toFixed(1)}`;
    ctx.fillText(signStr, lx, yCenter);
  });
}

export function renderIrrHistogramChart(canvas, bins = [], summary = {}) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !bins.length) return;
  const { ctx, width, height } = setup;

  const padLeft = 46;
  const padRight = 20;
  const padTop = 26;
  const padBottom = 34;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const maxFreq = Math.max(...bins.map((b) => b.frequency_pct), 12.0) * 1.15;
  const colW = plotW / bins.length;

  bins.forEach((b, idx) => {
    const xCenter = padLeft + idx * colW + colW * 0.5;
    const barW = Math.max(colW * 0.78, 6);
    const barH = (b.frequency_pct / maxFreq) * plotH;
    const yTop = padTop + plotH - barH;

    if (b.bin_center < 0) {
      ctx.fillStyle = COLORS.negative;
    } else if (b.bin_center >= 20.0) {
      ctx.fillStyle = COLORS.positive;
    } else {
      ctx.fillStyle = COLORS.cobalt;
    }
    ctx.fillRect(xCenter - barW / 2, yTop, barW, Math.max(barH, 2));

    if (idx % 3 === 0) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = '500 10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`%${b.bin_center.toFixed(0)}`, xCenter, height - 10);
    }
  });

  // Median IRR vertical marker line
  if (summary && summary.median_irr_pct !== undefined) {
    const minX = bins[0].bin_start;
    const maxX = bins[bins.length - 1].bin_end;
    const ratio = (summary.median_irr_pct - minX) / Math.max(maxX - minX, 1);
    const medX = padLeft + Math.min(Math.max(ratio, 0), 1) * plotW;

    ctx.strokeStyle = COLORS.goldBright;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(medX, padTop);
    ctx.lineTo(medX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = COLORS.goldBright;
    ctx.font = '600 11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`P50: %${summary.median_irr_pct.toFixed(1)}`, medX, padTop - 8);
  }
}

export function renderDecileLiftChart(canvas, decileRows = []) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !decileRows.length) return;
  const { ctx, width, height } = setup;

  const padLeft = 46;
  const padRight = 20;
  const padTop = 24;
  const padBottom = 32;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const maxLift = Math.max(...decileRows.map((d) => d.lift), 4.0) * 1.12;
  const colW = plotW / decileRows.length;

  const baseLineY = padTop + plotH - (1.0 / maxLift) * plotH;
  ctx.strokeStyle = 'rgba(244, 63, 94, 0.55)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padLeft, baseLineY);
  ctx.lineTo(width - padRight, baseLineY);
  ctx.stroke();
  ctx.setLineDash([]);

  decileRows.forEach((d, idx) => {
    const xCenter = padLeft + idx * colW + colW * 0.5;
    const barW = Math.min(colW * 0.58, 34);
    const barH = Math.max((d.lift / maxLift) * plotH, 3);
    const yTop = padTop + plotH - barH;

    ctx.fillStyle = idx === 0 ? COLORS.gold : COLORS.cobalt;
    ctx.fillRect(xCenter - barW / 2, yTop, barW, barH);

    ctx.fillStyle = COLORS.text;
    ctx.font = '600 10.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${d.lift.toFixed(1)}x`, xCenter, yTop - 6);

    ctx.fillStyle = COLORS.muted;
    ctx.font = '500 11px "JetBrains Mono", monospace';
    ctx.fillText(`D${d.decile}`, xCenter, height - 10);
  });
}

export function renderCalibrationCurveChart(canvas, relRows = []) {
  const setup = setupHiDPICanvas(canvas);
  if (!setup || !relRows.length) return;
  const { ctx, width, height } = setup;

  const padLeft = 52;
  const padRight = 24;
  const padTop = 20;
  const padBottom = 32;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const maxP = Math.min(
    Math.max(...relRows.map((r) => Math.max(r.mean_predicted_prob, r.observed_frequency)), 0.35) * 1.1,
    1.0
  );

  ctx.strokeStyle = COLORS.grid;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop + plotH);
  ctx.lineTo(padLeft + plotW, padTop);
  ctx.stroke();
  ctx.setLineDash([]);

  const groups = {};
  relRows.forEach((r) => {
    if (!groups[r.method_key]) groups[r.method_key] = [];
    groups[r.method_key].push(r);
  });

  const palette = {
    platt: COLORS.gold,
    isotonic: COLORS.positive,
    uncalibrated: COLORS.cobalt,
  };

  Object.entries(groups).forEach(([mKey, pts]) => {
    ctx.strokeStyle = palette[mKey] || COLORS.cobalt;
    ctx.fillStyle = palette[mKey] || COLORS.cobalt;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = padLeft + (Math.min(p.mean_predicted_prob, maxP) / maxP) * plotW;
      const y = padTop + plotH - (Math.min(p.observed_frequency, maxP) / maxP) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    pts.forEach((p) => {
      const x = padLeft + (Math.min(p.mean_predicted_prob, maxP) / maxP) * plotW;
      const y = padTop + plotH - (Math.min(p.observed_frequency, maxP) / maxP) * plotH;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}
