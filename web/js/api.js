/**
 * Origon REST API Client Layer (supports lang = 'tr' | 'en')
 */

export async function fetchMetadata(lang = 'tr') {
  const res = await fetch(`/api/v1/metadata?lang=${encodeURIComponent(lang)}`);
  if (!res.ok) throw new Error('Failed to load metadata');
  return res.json();
}

export async function fetchOverview(lang = 'tr') {
  const res = await fetch(`/api/v1/overview?lang=${encodeURIComponent(lang)}`);
  if (!res.ok) throw new Error('Failed to load overview');
  return res.json();
}

export async function fetchTargets(params = {}, lang = 'tr') {
  const qs = new URLSearchParams();
  qs.set('lang', lang);
  if (params.regions && params.regions.length) qs.set('regions', params.regions.join(','));
  if (params.sectors && params.sectors.length) qs.set('sectors', params.sectors.join(','));
  if (params.sizes && params.sizes.length) qs.set('sizes', params.sizes.join(','));
  if (params.onlyLive) qs.set('only_live', 'true');
  if (params.minProb !== undefined) qs.set('min_prob', String(params.minProb));
  if (params.maxLeverage !== undefined) qs.set('max_leverage', String(params.maxLeverage));
  if (params.q) qs.set('q', params.q);
  if (params.sortBy) qs.set('sort_by', params.sortBy);
  if (params.sortDir) qs.set('sort_dir', params.sortDir);
  qs.set('limit', String(params.limit || 250));

  const res = await fetch(`/api/v1/targets?${qs.toString()}`);
  if (!res.ok) throw new Error('Failed to load targets');
  return res.json();
}

export async function fetchTargetDetail(ticker, lang = 'tr') {
  const res = await fetch(`/api/v1/targets/${encodeURIComponent(ticker)}?lang=${encodeURIComponent(lang)}`);
  if (!res.ok) throw new Error(`Failed to load target detail: ${ticker}`);
  return res.json();
}

export async function fetchBacktest(lang = 'tr') {
  const res = await fetch(`/api/v1/backtest?lang=${encodeURIComponent(lang)}`);
  if (!res.ok) throw new Error('Failed to load backtest metrics');
  return res.json();
}

export async function triggerPipelineRefresh() {
  const res = await fetch('/api/v1/pipeline/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ use_live_yfinance: true }),
  });
  if (!res.ok) throw new Error('Pipeline refresh failed');
  return res.json();
}
