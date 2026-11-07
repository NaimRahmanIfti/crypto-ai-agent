import { Scene3D } from './scene3d.js?v=9';
import { candleChart, equityChart, calibrationChart } from './charts.js?v=9';

const $ = id => document.getElementById(id);
const state = { asset: 'crypto', symbol: 'BTC', interval: '1d', horizon: 1, universe: null, last: null, forming: null, livePrice: null };
const pct = (v, d = 1) => v == null ? '—' : `${(v * 100).toFixed(d)}%`;
const num = (v, d = 2) => v == null ? '—' : v.toFixed(d);
const money = v => v == null ? '—' : v >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 2 })
  : v >= 1 ? v.toFixed(2) : v.toPrecision(4);
const SIG = { up: '#34C79A', down: '#E5677A', amber: '#D9A75B', accent: '#6E8BFF' };
const esc = t => String(t ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
const vTone = v => /BUY/.test(v) ? 'up' : /SELL|AVOID|REDUCE/.test(v) ? 'down' : 'amber';
const toneCls = t => t === 'bull' ? 'up' : t === 'bear' ? 'down' : 'amber';
const safeLink = u => /^https?:\/\//.test(u || '') ? u : '#';
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------------------------------------------------------------- 3D scene
let scene = null;
const sceneReady = new Scene3D().init($('bg3d')).then(s => { scene = s; if (state.last) paint3d(state.last); return s; })
  .catch(e => { console.warn('3D disabled:', e); return null; });

async function api(path, timeoutMs = 90000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  let r;
  try { r = await fetch(path, { signal: ctl.signal }); }
  catch (e) { throw new Error(e.name === 'AbortError' ? `The server took longer than ${timeoutMs / 1000}s to answer. Data sources may be slow; try again.` : 'Cannot reach the server.'); }
  finally { clearTimeout(timer); }
  let js;
  try { js = await r.json(); } catch { throw new Error(`Server returned ${r.status} ${r.statusText}`); }
  if (!r.ok) throw new Error(js.error || r.statusText);
  return js;
}

// ---------------------------------------------------------------- boot sequence
const intro = {
  t0: performance.now(), done: false,
  step(frac, text) { $('introBar').style.width = `${Math.round(frac * 100)}%`; if (text) $('introLine').textContent = text; },
  async finish() {
    if (this.done) return;
    this.done = true;
    const minShow = reduceMotion ? 0 : 2600;  // let the fly-in land even on a fast cache hit
    const wait = Math.max(0, minShow - (performance.now() - this.t0));
    await new Promise(r => setTimeout(r, wait));
    this.step(1, 'Ready');
    $('intro').classList.add('done');
    document.body.classList.remove('booting');
    document.body.classList.add('revealed');
    if (state.last) render(state.last);  // re-measure canvases now that they are visible
  },
};
$('introSkip').onclick = () => intro.finish();
addEventListener('keydown', e => { if (e.key === 'Escape') intro.finish(); });

// ---------------------------------------------------------------- status helpers
function setStatus(text, cls = '') { $('status').className = `status ${cls}`; $('status').querySelector('span').textContent = text; }
function loading(on, text = 'Training models') {
  $('loader').classList.toggle('hide', !on); $('loaderText').textContent = text;
  document.querySelectorAll('main .card').forEach(c => c.classList.toggle('loading', on && !!state.last));
}
function banner(html) { $('banner').hidden = !html; $('banner').innerHTML = html || ''; }

async function checkServer() {
  try {
    const v = await api('/api/version');
    if (v.stale) banner('The server is running out-of-date code, so some panels may break. Stop it and start it again: <code>lsof -ti tcp:8000 | xargs kill</code> then <code>python3 server.py</code>');
    else if (state.universe?.demo) banner('Demo mode: every price on this page is synthetic. Start the server without <code>QUANT_DEMO=1</code> for real markets.');
    else banner('');
  } catch {
    banner('This page is newer than the server it is talking to. Restart the server: <code>lsof -ti tcp:8000 | xargs kill</code> then <code>python3 server.py</code>');
  }
}

function renderControls() {
  const u = state.universe;
  document.querySelectorAll('#assetSeg button').forEach(b => b.classList.toggle('on', b.dataset.asset === state.asset));
  // Popular markets as chips; a searched market gets its own chip so it stays visible.
  const chips = u[state.asset].includes(state.symbol) ? u[state.asset] : [...u[state.asset], state.symbol];
  $('symbols').innerHTML = chips.map(s => `<button class="${s === state.symbol ? 'on' : ''}" data-s="${esc(s)}">${esc(s)}</button>`).join('');
  $('symList').innerHTML = (u.all?.[state.asset] || u[state.asset]).map(s => `<option value="${esc(s)}">`).join('');
  $('customSym').placeholder = state.asset === 'crypto' ? `Search ${u.all?.crypto.length ?? ''} coins, or any Binance coin` : `Search ${u.all?.stock.length ?? ''} stocks and ETFs, or any US ticker`;
  $('interval').innerHTML = u.intervals[state.asset].map(i => `<option ${i === state.interval ? 'selected' : ''}>${i}</option>`).join('');
  $('horizon').value = state.horizon;
}

// ---------------------------------------------------------------- live price (display only)
// The model and backtests only use completed candles, so a half-formed bar never leaks into a signal.
let liveTimer = null;
function startLive() {
  clearInterval(liveTimer);
  $('live').textContent = '';
  state.livePrice = null;
  pollLive();
  liveTimer = setInterval(pollLive, state.asset === 'crypto' ? 5000 : 15000);
}

const drawChart = (bars, proba) => candleChart($('candles'), bars, proba, (b, p) => {
  $('hoverInfo').textContent = b ? `${new Date(b.t * 1000).toISOString().slice(0, 16).replace('T', ' ')}  O ${money(b.o)}  H ${money(b.h)}  L ${money(b.l)}  C ${money(b.c)}${p != null ? `  P(up) ${pct(p)}` : ''}` : '';
});

async function pollLive() {
  const d = state.last;
  if (!d || document.hidden) return;
  try {
    const q = await api(`/api/quote?symbol=${d.symbol}&asset=${d.asset_class}`);
    if (state.last !== d || !(q.price > 0)) return;
    const prev = state.livePrice;
    state.livePrice = q.price;
    const el = $('price');
    el.textContent = money(q.price);
    if (prev != null && q.price !== prev) {
      el.classList.remove('flash-up', 'flash-down'); void el.offsetWidth;
      el.classList.add(q.price > prev ? 'flash-up' : 'flash-down');
    }
    const cp = q.change_pct ?? 0;
    $('chg').textContent = `${cp >= 0 ? '+' : ''}${cp.toFixed(2)}% ${d.asset_class === 'crypto' ? '24h' : 'today'}`;
    $('chg').className = `chg ${cp >= 0 ? 'up' : 'down'}`;
    $('live').innerHTML = `<i class="dot"></i>Live from ${esc(q.source)}${q.market ? `, ${esc(q.market)}` : ''}, ${esc(q.as_of || new Date().toLocaleTimeString())}`;
    const last = d.ohlcv[d.ohlcv.length - 1];
    const step = d.ohlcv.length > 1 ? last.t - d.ohlcv[d.ohlcv.length - 2].t : 86400;
    const f = state.forming ?? { t: last.t + step, o: last.c, h: Math.max(last.c, q.price), l: Math.min(last.c, q.price), c: q.price, v: 0 };
    f.h = Math.max(f.h, q.price); f.l = Math.min(f.l, q.price); f.c = q.price;
    state.forming = f;
    drawChart([...d.ohlcv, f], d.proba_history);
  } catch (e) {
    const offline = /No internet/i.test(e.message);
    if (offline) setStatus('offline, retrying', 'err');
    $('live').innerHTML = `<i class="dot off"></i>${offline ? 'Offline: no internet connection. Retrying automatically' : `Live price unavailable (${esc(e.message).slice(0, 90)})`}. Showing the last close.`;
  }
}

// ---------------------------------------------------------------- main load
let loadSeq = 0;
async function load() {
  const seq = ++loadSeq;
  renderControls();
  loading(true, `Training on ${state.symbol} ${state.interval}`);
  setStatus('computing');
  if (!intro.done) intro.step(0.45, `Training walk-forward models on ${state.symbol}`);
  try {
    const d = await api(`/api/analyze?symbol=${state.symbol}&asset=${state.asset}&interval=${state.interval}&horizon=${state.horizon}`);
    if (seq !== loadSeq) return;  // a newer request superseded this one
    state.last = d;
    state.forming = null;
    if (!intro.done) intro.step(0.85, 'Reading the chart, the news and the economy');
    render(d);
    setStatus(`${d.provider}, ${d.compute_seconds}s`, 'live');
    startLive();
    startMtfLive();
  } catch (e) {
    if (seq !== loadSeq) return;
    setStatus('error', 'err');
    state.last = null;
    showError(e.message);
  } finally {
    if (seq === loadSeq) { loading(false); intro.finish(); }
  }
  loadMacro();
}

// Clear every panel so a failed load never leaves another market's numbers on screen.
function showError(msg) {
  clearInterval(liveTimer); clearInterval(mtfTimer); $('live').textContent = ''; state.forming = null;
  $('heroCard').style.setProperty('--sig', SIG.down);
  $('sym').textContent = `${state.symbol}, ${state.asset === 'crypto' ? 'crypto' : 'stock'}, ${state.interval}`;
  ['price', 'chg', 'meta', 'hoverInfo', 'summary', 'invalid', 'newsNote', 'deskNote', 'sessTag', 'btTag'].forEach(id => $(id).textContent = '');
  $('pUp').textContent = '—'; $('gaugeArc').style.strokeDashoffset = 314;
  $('action').textContent = 'No data';
  const hint = /429|rate limit/i.test(msg) ? 'The data provider is rate-limiting this connection. Wait a few minutes or switch market.'
    : /unavailable|HTTP/i.test(msg) ? 'Run python3 scripts/diagnose.py to see which data source is failing.' : '';
  $('reason').innerHTML = `<span class="err-box"><b>${esc(msg)}</b>${esc(hint)}</span>`;
  ['levels', 'valKpis', 'confBars', 'btStats', 'regime', 'features', 'horizons', 'scenarios', 'bullF', 'bearF', 'neutF', 'levelsMap', 'news', 'newsThemes',
    'mtfTable', 'learnKinds', 'learnMisses', 'learnCal', 'learnSkill'].forEach(id => $(id).innerHTML = '');
  ['mtfVerdict', 'learnSummary'].forEach(id => $(id).textContent = '');
  $('verdict').textContent = '—';
  ['candles', 'calib', 'equity'].forEach(id => { const c = $(id); c.getContext('2d').clearRect(0, 0, c.width, c.height); });
}

function paint3d(d) {
  if (!scene) return;
  scene.setCandles(d.ohlcv);
  scene.setSignal(d.signal.tone, d.signal.exposure);
}

// ---------------------------------------------------------------- render
function render(d) {
  const s = d.signal, v = d.validation;
  const tone = s.tone === 'up' ? 'up' : 'amber';
  $('heroCard').style.setProperty('--sig', SIG[tone]);
  $('sym').textContent = `${d.symbol}, ${d.asset_class === 'crypto' ? 'crypto' : 'stock'}, ${d.interval} candles, ${d.horizon}-bar horizon`;
  if (state.livePrice == null) {
    $('price').textContent = money(d.last_price);
    $('chg').textContent = `${d.change_pct >= 0 ? '+' : ''}${d.change_pct.toFixed(2)}% last bar`;
    $('chg').className = `chg ${d.change_pct >= 0 ? 'up' : 'down'}`;
  }
  const age = d.data_age_seconds ?? 0;
  const fresh = age < 3600 ? 'up to date' : age < 86400 ? `${Math.round(age / 3600)}h behind` : `${Math.round(age / 86400)} days behind`;
  $('meta').innerHTML = `${d.bars.toLocaleString()} candles from ${d.start.slice(0, 10)} to ${d.end.slice(0, 10)}, from ${esc(d.provider)} ` +
    `<span class="fresh ${age < 3600 ? 'ok' : 'stale'}">(${fresh})</span>`;
  $('pUp').textContent = pct(s.exposure, 0);
  $('gaugeArc').style.strokeDashoffset = 314 * (1 - s.exposure);
  $('action').textContent = s.action;
  $('reason').textContent = s.reason;
  const lv = [['Trend score', `${Math.round(s.trend_score * 3)} of 3 lookbacks`], ['Model P(up), next bar', `${pct(s.p_up)}${s.ml_validated ? ' (validated)' : ''}`],
    ['Entry', money(s.entry)], ['Trailing stop', `${money(s.stop)} (−${pct(s.risk_to_stop_pct)})`],
    ['Scale out at 2R', money(s.take_profit_1)], ['Scale out at 4R', money(s.take_profit_2)],
    ['Volatility', `${pct(s.realized_vol, 0)} now, ${pct(s.target_vol, 0)} target`], ['Account risk if stopped', pct(s.account_risk_pct, 2)]];
  $('levels').innerHTML = lv.map(([k, x]) => `<div class="lv"><small>${k}</small><b>${x}</b></div>`).join('');

  drawChart(state.forming ? [...d.ohlcv, state.forming] : d.ohlcv, d.proba_history);

  const edge = v.accuracy - v.baseline_accuracy;
  $('valKpis').innerHTML = [
    ['Accuracy', pct(v.accuracy), `base rate ${pct(v.baseline_accuracy)}`, edge > 0 ? 'up' : 'down'],
    ['AUC', num(v.auc, 3), '0.5 is random', v.auc > 0.52 ? 'up' : ''],
    ['p-value', v.p_value_vs_coinflip < 0.001 ? '<0.001' : num(v.p_value_vs_coinflip, 3), 'vs coin flip', v.p_value_vs_coinflip < 0.05 ? 'up' : 'amber'],
    ['Information coeff.', num(v.ic, 3), `${v.n.toLocaleString()} test bars`, v.ic > 0 ? 'up' : 'down'],
  ].map(([k, x, sub, c]) => `<div class="kpi"><small>${k}</small><b class="${c}">${x}</b><i>${sub}</i></div>`).join('');
  $('confBars').innerHTML = `<div class="bar"><span>Naive momentum</span><div class="track"><div class="fill" style="width:${d.momentum_baseline_accuracy * 100}%"></div><div class="mark" style="left:50%"></div></div><span>${pct(d.momentum_baseline_accuracy)}</span></div>` +
    v.confident.map(c => `<div class="bar"><span>Confident ≥${(c.threshold * 100).toFixed(0)}%, ${pct(c.coverage, 0)} of bars</span><div class="track"><div class="fill" style="width:${c.accuracy * 100}%"></div><div class="mark" style="left:50%"></div></div><span>${pct(c.accuracy)}</span></div>`).join('');
  calibrationChart($('calib'), v.calibration);

  const b = d.backtest, st = b.strategy, bh = b.buy_hold;
  $('btTag').textContent = `trend core, long or flat, ${b.config.cost_bps} bps costs, ${pct(b.config.target_vol, 0)} vol target`;
  equityChart($('equity'), b.equity);
  const rows = [['CAGR', 'cagr', pct], ['Sharpe', 'sharpe', num], ['Sortino', 'sortino', num], ['Max drawdown', 'max_drawdown', x => pct(x, 0)],
    ['Volatility', 'volatility', x => pct(x, 0)], ['Hit rate', 'hit_rate', pct], ['Profit factor', 'profit_factor', num],
    ['Time in market', 'exposure', x => pct(x, 0)], ['Chance Sharpe > 0', 'psr', x => pct(x, 0)]];
  $('btStats').innerHTML = `<tr><td></td><td style="color:${SIG.accent}">Trend core</td><td>Model only</td><td>Blend</td><td>Buy and hold</td></tr>` +
    rows.map(([k, key, f]) => `<tr><td>${k}</td><td>${f(st[key])}</td><td>${f(b.ml?.[key])}</td><td>${f(b.blend?.[key])}</td><td>${f(bh[key])}</td></tr>`).join('');

  const r = d.regime;
  $('regime').innerHTML = [['Trend', r.trend, r.trend === 'uptrend' ? 'up' : r.trend === 'downtrend' ? 'down' : 'amber'],
    ['Volatility', `${r.volatility}, p${Math.round(r.vol_percentile * 100)}`, r.volatility === 'high' ? 'down' : '']]
    .map(([k, x, c]) => `<div class="kpi"><small>${k}</small><b class="${c}" style="font-size:16px">${x}</b></div>`).join('');
  const maxI = Math.max(...d.features.map(f => f.importance));
  $('features').innerHTML = d.features.slice(0, 10).map(f => `<div class="bar"><span>${f.feature.replace(/_/g, ' ')}</span><div class="track"><div class="fill" style="width:${f.importance / maxI * 100}%"></div></div><span>${pct(f.importance, 1)}</span></div>`).join('');

  renderDesk(d.analysis, d.last_price);
  renderMtf(d.mtf);
  renderLearning(d);
  paint3d(d);
}

// First sentence as the one-line takeaway; the rest opens on click. ('. ' keeps decimals like 3.1 intact.)
const firstSentence = t => { const i = (t || '').search(/\.\s/); return i < 0 ? (t || '') : t.slice(0, i + 1); };
const restOf = t => (t || '').slice(firstSentence(t).length).trim();
const CHIP = { bull: 'Bullish', bear: 'Bearish', neutral: 'Neutral', unknown: 'No data' };
const barHtml = s => {
  const v = Math.max(-1, Math.min(1, s || 0)), w = Math.abs(v) * 50, left = v >= 0 ? 50 : 50 - w;
  return `<span class="fx-bar" title="strength ${v >= 0 ? '+' : ''}${v.toFixed(2)}"><i style="left:${left}%;width:${w}%;background:${v >= 0 ? SIG.up : SIG.down}"></i></span>`;
};
const factorHtml = (f, chips = CHIP) => {
  const rest = restOf(f.text);
  return `<details class="fx ${f.tone}"><summary><span class="fx-name">${esc(f.name)}</span>${barHtml(f.score)}
    <span class="fx-head">${esc(firstSentence(f.text))}</span><span class="chip ${f.tone}">${chips[f.tone] || f.tone}</span></summary>
    ${rest ? `<p>${esc(rest)}</p>` : ''}</details>`;
};
const ECON_CHIP = { bull: 'Supportive', bear: 'Concerning', neutral: 'Neutral', unknown: 'No data' };
const takeawayHtml = items => items.filter(Boolean).map(([tone, text, sub]) =>
  `<div class="tk ${tone}"><i>${tone === 'bull' ? '▲' : tone === 'bear' ? '▼' : '●'}</i><div>${esc(text)}${sub ? `<small>${esc(sub)}</small>` : ''}</div></div>`).join('');
const newsHtml = items => items.length ? items.map(n => {
  const ago = n.t ? Math.max(1, Math.round((Date.now() / 1000 - n.t) / 3600)) : null;
  return `<a href="${esc(safeLink(n.link))}" target="_blank" rel="noopener noreferrer"><i class="${n.sentiment > 0 ? 'p' : n.sentiment < 0 ? 'n' : ''}"></i>
    <span>${esc(n.title)}<small>${esc(n.source)}${n.category ? `, ${esc(n.category.toLowerCase())}` : ''}${ago ? `, ${ago < 48 ? `${ago}h` : `${Math.round(ago / 24)}d`} ago` : ''}</small></span></a>`;
}).join('') : '<div class="empty">No headlines right now.</div>';

function renderDesk(a, price) {
  $('desk').style.setProperty('--dk', SIG[vTone(a.verdict)]);
  $('verdict').textContent = a.verdict;
  $('meter').style.left = `${Math.max(2, Math.min(98, 50 + a.score * 100))}%`;
  $('summary').textContent = a.summary;
  $('sessTag').textContent = `${a.session.label}, ${a.session.next_event}`;
  // The three things to know: strongest reason for, strongest against, and where the view breaks.
  const strongest = t => a.factors.filter(f => f.tone === t).sort((x, y) => Math.abs(y.score) - Math.abs(x.score))[0];
  const up = strongest('bull'), dn = strongest('bear');
  $('whyTakeaways').innerHTML = takeawayHtml([
    up && ['bull', `${up.name}: ${firstSentence(up.text)}`, 'strongest reason to be bullish'],
    dn && ['bear', `${dn.name}: ${firstSentence(dn.text)}`, 'strongest reason to be cautious'],
    ['neutral', firstSentence(a.invalidation), 'what would prove this view wrong'],
  ]);
  $('horizons').innerHTML = a.horizons.map(h => `<details class="hz" style="--hz:${SIG[vTone(h.verdict)]}">
      <summary><small>${esc(h.title)}</small><b>${esc(h.verdict)}</b>
      ${h.range_68 ? `<div class="rng">Expected range ${money(h.range_68[0])} to ${money(h.range_68[1])}, model P(up) ${pct(h.p_up)}</div>` : ''}
      <span class="more-hint">Why</span></summary><p>${esc(h.text)}</p></details>`).join('');
  const sc = a.scenarios || [];
  const scTone = n => /Bull/.test(n) ? SIG.up : /Bear/.test(n) ? SIG.down : SIG.accent;
  $('scenarios').innerHTML = sc.map(x => `<div class="scn" style="--sc:${scTone(x.name)}">
      <div class="scn-h"><b>${esc(x.name)}</b><span class="scn-p">${x.p}%</span></div>
      <div class="scn-bar"><div style="width:${x.p}%"></div></div>
      <p>${esc(x.text)}</p><small>Target ${money(x.target)}. Invalidated by ${esc(x.invalidation.replace(/\.$/, '').toLowerCase())}.</small></div>`).join('');
  const by = t => a.factors.filter(f => f.tone === t).sort((x, y) => Math.abs(y.score) - Math.abs(x.score));
  // Arrow wrappers matter: .map(factorHtml) would pass the index as the chip-label argument.
  $('bullF').innerHTML = by('bull').map(f => factorHtml(f)).join('') || '<div class="empty">None right now.</div>';
  $('bearF').innerHTML = by('bear').map(f => factorHtml(f)).join('') || '<div class="empty">None right now.</div>';
  $('neutF').innerHTML = by('neutral').map(f => factorHtml(f)).join('');
  const L = a.levels, touch = r => `${r.touches} touch${r.touches > 1 ? 'es' : ''}`;
  const rows = [
    ...L.resistance.slice().reverse().map(r => ['r', `Resistance, ${touch(r)}`, r.price]),
    ['p', 'Last close', price],
    ...L.support.map(r => ['s', `Support, ${touch(r)}`, r.price]),
    ['x', '52-week high', L.high_52w], ['x', '52-week low', L.low_52w]];
  $('levelsMap').innerHTML = rows.map(([c, k, v]) => `<div class="lvl ${c}"><span>${k}</span><span>${money(v)}<small>${c === 'p' ? '' : `${((v / price - 1) * 100).toFixed(1)}%`}</small></span></div>`).join('');
  $('invalid').textContent = a.invalidation;
  $('newsNote').textContent = a.news.note ? `(${a.news.note})` : '';
  $('newsThemes').innerHTML = (a.news.themes || []).map(t => `<span class="theme ${t.sentiment > 0.05 ? 'bull' : t.sentiment < -0.05 ? 'bear' : ''}">${esc(t.category)} ${t.n}</span>`).join('');
  $('news').innerHTML = newsHtml(a.news.items || []);
  $('deskNote').textContent = a.disclaimer;
}

// ---------------------------------------------------------------- world economy
// The server answers instantly with whatever is ready; poll until the full build lands.
let macroLoadedFor = null, macroTimer = null, macroTries = 0;
async function loadMacro() {
  const key = state.asset;
  if (macroLoadedFor === key) return;
  macroLoadedFor = key;
  clearTimeout(macroTimer);
  macroTries = 0;
  pollMacro(key);
}

async function pollMacro(key) {
  if (key !== state.asset) return;
  try {
    const [m, c] = await Promise.all([api('/api/macro', 20000),
      key === 'crypto' ? api('/api/crypto_regime', 20000).catch(() => null) : null]);
    renderMacro(m, c);
    const pending = m.building || (key === 'crypto' && c && c.building);
    if (pending && ++macroTries < 60) {
      $('macroTag').textContent = `collecting data${'.'.repeat(1 + (macroTries % 3))}`;
      macroTimer = setTimeout(() => pollMacro(key), 4000);
    }
  } catch (e) {
    macroLoadedFor = null;
    $('macroTag').textContent = 'unavailable';
    $('macroSummary').textContent = `World-economy data could not be loaded (${e.message}). It retries on the next market switch.`;
  }
}

function renderMacro(m, c) {
  const rgTone = { Broadening: SIG.up, Concentration: SIG.amber, Contraction: SIG.down, Inflationary: SIG.down, Transitional: SIG.accent };
  if (!m.ok) {
    $('macroRegime').innerHTML = m.building ? '<span class="ring inline"></span>' : '—';
    $('macroSummary').textContent = m.note || 'Macro data unavailable.';
  } else {
    $('macroCard').style.setProperty('--rg', rgTone[m.regime] || SIG.accent);
    $('macroRegime').innerHTML = `${esc(m.regime)}<small>Cross-asset score ${m.score_100.toFixed(0)} of 100</small>`;
    $('macroSummary').textContent = m.summary;
  }
  $('macroTag').textContent = m.as_of ? `updated ${m.as_of}` : 'world economy';
  const wait = m.building ? '<div class="empty">Downloading, usually under a minute on first start.</div>' : '';
  const comps = (m.components || []).map(x => ({ ...x, tone: x.ok ? x.tone : 'unknown' }));
  $('macroComps').innerHTML = comps.map(x => factorHtml(x, ECON_CHIP)).join('') || wait || '<div class="empty">No cross-asset data.</div>';
  $('econ').innerHTML = (m.economy || []).map(e => {
    const rest = restOf(e.text);
    return `<details class="eco ${e.tone}"><summary><div class="eco-top"><small>${esc(e.name)}</small><span class="chip ${e.tone}">${ECON_CHIP[e.tone] || 'Neutral'}</span></div>
      <b>${esc(e.value ?? '—')}</b><div class="eco-head">${esc(firstSentence(e.text))}</div></summary>${rest ? `<p>${esc(rest)}</p>` : ''}</details>`;
  }).join('') || wait;
  // Key takeaways: the regime's posture, plus the most supportive and most worrying signals.
  if (m.ok) {
    const ranked = [...comps.filter(x => x.ok), ...(m.economy || []).filter(e => e.value).map(e => ({ name: e.name, text: e.text, tone: e.tone,
      score: e.tone === 'bull' ? 0.5 : e.tone === 'bear' ? -0.5 : 0 }))];
    const best = ranked.filter(x => x.tone === 'bull').sort((a, b) => b.score - a.score)[0];
    const worst = ranked.filter(x => x.tone === 'bear').sort((a, b) => a.score - b.score)[0];
    $('econTakeaways').innerHTML = takeawayHtml([
      ['neutral', firstSentence(m.posture), `${m.regime} regime: what it means for positioning`],
      best && ['bull', `${best.name}: ${firstSentence(best.text)}`, 'most supportive signal'],
      worst && ['bear', `${worst.name}: ${firstSentence(worst.text)}`, 'biggest concern'],
    ]);
  } else {
    $('econTakeaways').innerHTML = '';
  }
  $('calendar').innerHTML = (m.calendar || []).map(e => {
    const dt = new Date(`${e.date}T12:00:00Z`);
    return `<div class="cal"><b>${dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</b><div>${esc(e.event)}<small>${esc(e.why)}</small></div></div>`;
  }).join('');
  $('macroNews').innerHTML = newsHtml((m.news && m.news.items) || []);
  const wrap = $('cryptoRegimeWrap');
  if (c && c.ok) {
    wrap.hidden = false;
    $('cryptoRegimeScore').textContent = `(${c.score.toFixed(0)} of 100, ${c.zone.replace('_', '-').toLowerCase()})`;
    $('cryptoComps').innerHTML = c.components.map(x => factorHtml({ name: x.name, score: (x.score - 50) / 50, text: x.text,
      tone: x.score >= 60 ? 'bull' : x.score < 40 ? 'bear' : 'neutral' })).join('');
  } else wrap.hidden = true;
}

// ---------------------------------------------------------------- every timeframe
const lblTone = l => /LONG/.test(l) ? 'up' : /SHORT/.test(l) ? 'down' : 'amber';
const sbar = s => {
  const w = Math.abs(s) * 50, left = s >= 0 ? 50 : 50 - w;
  return `<span class="sbar"><i style="left:${left}%;width:${w}%;background:${s >= 0 ? SIG.up : SIG.down}"></i></span>`;
};
function renderMtf(m) {
  if (!m) { $('mtfVerdict').textContent = 'Multi-timeframe data unavailable right now.'; $('mtfTable').innerHTML = ''; return; }
  $('mtfVerdict').textContent = m.verdict;
  $('mtfTag').textContent = `confluence ${m.confluence >= 0 ? '+' : ''}${m.confluence.toFixed(2)}, ${m.longs} bullish, ${m.shorts} bearish`;
  const toneVar = l => SIG[lblTone(l)] || SIG.amber;
  $('mtfTable').innerHTML = m.rows.map(r => r.ok ? `<div class="tf" style="--tfc:${toneVar(r.label)}">
      <div class="tf-h"><b>${esc(r.tf)}</b><small>${esc(r.style)}</small></div>
      <div class="tf-l">${esc(r.label)}</div>${barHtml(r.score)}
      <div class="tf-row"><span>Trend (context)</span><span>${esc((r.trend_label || '').toLowerCase())}</span></div>
      <div class="tf-row"><span>RSI</span><span>${r.rsi.toFixed(0)}</span></div>
      <div class="tf-row"><span>Recent move</span><span class="${r.ret_1m >= 0 ? 'up' : 'down'}">${pct(r.ret_1m)}</span></div>
      <div class="tf-plan">${r.plan.side === 'WAIT' ? esc(r.plan.text)
        : `<b class="${r.plan.side === 'LONG' ? 'up' : 'down'}">${r.plan.side}</b> at ${money(r.plan.entry)}<br>stop ${money(r.plan.stop)}, targets ${money(r.plan.target_1)} and ${money(r.plan.target_2)}
           ${r.evidence ? `<small style="display:block;color:var(--muted);margin-top:6px">Evidence: ${esc(r.evidence)}</small>` : ''}`}</div></div>`
    : `<div class="tf na"><div class="tf-h"><b>${esc(r.tf)}</b><small>${esc(r.style)}</small></div><div class="tf-plan">Not available right now${/stock|Yahoo|429/i.test(r.error) ? ' (stock intraday data only exists during market hours)' : ''}.</div></div>`).join('');
}

// 1m-30m signals go stale fast: refresh the timeframe table every minute on its own.
let mtfTimer = null;
function startMtfLive() {
  clearInterval(mtfTimer);
  const d = state.last;
  if (!d) return;
  mtfTimer = setInterval(async () => {
    if (document.hidden || state.last !== d) return;
    try {
      const m = await api(`/api/mtf?symbol=${d.symbol}&asset=${d.asset_class}`, 30000);
      if (state.last === d) { d.mtf = m; renderMtf(m); $('mtfTag').textContent += `, updated ${new Date().toLocaleTimeString()}`; }
    } catch { /* keep the last table; the next tick retries */ }
  }, 60000);
}

// ---------------------------------------------------------------- track record and learning
function renderLearning(d) {
  const c = d.calibration || {};
  $('learnCal').innerHTML = c.applied ? [
    ['Raw model P(up)', pct(c.raw), 'before calibration'], ['Calibrated P(up)', pct(c.calibrated), 'what the bot now uses'],
    ['Error before', num(c.brier_raw, 3), 'Brier score'], ['Error after', num(c.brier_cal, 3), `${c.n.toLocaleString()} past predictions`],
  ].map(([k, v, s]) => `<div class="kpi"><small>${k}</small><b>${v}</b><i>${s}</i></div>`).join('')
    : `<div class="empty">Not calibrated yet: ${esc(c.reason || 'not enough history')}.</div>`;
  const j = d.journal;
  if (!j) { $('learnSummary').textContent = 'The prediction journal is unavailable.'; return; }
  const a = j.asset_class, mk = j.market;
  const cls = d.asset_class === 'crypto' ? 'crypto' : 'stock';
  $('learnTag').textContent = `${a.total.toLocaleString()} ${cls} calls logged, ${a.resolved.toLocaleString()} checked, ${a.open.toLocaleString()} waiting`;
  $('learnSummary').textContent = a.resolved === 0
    ? `Learning has started. Every signal the bot gives is written to its journal with the price at that moment, then checked automatically when it comes due (next bar, 5 bars, 10 bars, by timeframe). Hit rates appear here as soon as the first calls resolve, and factor weights start adapting after ${40} checked swing calls.`
    : `${a.learning_active ? 'Learning is active: factor weights are being adjusted from' : 'Collecting evidence:'} ${a.resolved.toLocaleString()} checked ${cls} calls. ` +
      `For ${d.symbol}: ${mk.resolved} checked, ${mk.open} waiting. Hit rates below are live, out-of-sample results, not backtests.`;
  $('learnKinds').innerHTML = a.kinds.length ? a.kinds.map(k => `<div class="bar"><span>${esc(k.name)} (${k.n})</span>
      <div class="track"><div class="fill" style="width:${k.hit_rate * 100}%"></div><div class="mark" style="left:50%"></div></div>
      <span class="${k.hit_rate > 0.5 ? 'up' : 'down'}">${pct(k.hit_rate, 0)}</span></div>`).join('')
    : '<div class="empty">No calls have come due yet.</div>';
  $('learnMisses').innerHTML = a.misses.length ? a.misses.map(m => factorHtml({ name: `${m.symbol}, ${m.kind}`, score: m.ret, tone: 'bear',
      text: m.lesson })).join('') : '<div class="empty">No mistakes recorded yet.</div>';
  const fs = a.factor_skill || [];
  $('learnSkill').innerHTML = fs.length ? fs.map(f => `<div class="bar"><span>${esc(f.factor)} (n=${f.n})</span>
      <div class="track"><div class="fill" style="width:${Math.min(100, Math.abs(f.ic) * 200)}%;background:${f.ic >= 0 ? SIG.up : SIG.down}"></div></div>
      <span>×${f.multiplier.toFixed(2)}</span></div>`).join('')
    : '<div class="empty">Factor skill appears after 10 checked swing calls; weights adapt after 40.</div>';
}

// ---------------------------------------------------------------- all markets screener
const mk = { asset: null, offset: 0, limit: 50, sort: 'rank', desc: false, q: '', signal: '', sector: '', timer: null, total: 0 };
const big = v => v == null ? '—' : v >= 1e12 ? `${(v / 1e12).toFixed(2)}T` : v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(0)}K` : v.toFixed(0);
async function loadMarkets(reset = false) {
  if (mk.asset !== state.asset) { Object.assign(mk, { asset: state.asset, offset: 0, sector: '' }); reset = true; }
  if (reset) mk.offset = 0;
  clearTimeout(mk.timer);
  const p = new URLSearchParams({ asset: mk.asset, offset: mk.offset, limit: mk.limit, sort: mk.sort, desc: mk.desc ? 1 : 0,
    q: mk.q, signal: mk.signal, sector: mk.sector });
  try {
    const r = await api(`/api/markets?${p}`, 40000);
    if (r.asset !== state.asset) return;
    mk.total = r.matches;
    const isStock = r.asset === 'stock';
    $('marketsTitle').textContent = isStock ? `All US stocks (${r.total.toLocaleString()})` : `All crypto (${r.total.toLocaleString()} coins)`;
    $('marketsTag').innerHTML = `<span class="fresh ${r.age_seconds < 120 ? 'ok' : 'stale'}"><i class="dot"></i>prices ${r.age_seconds < 60 ? 'live' : `${Math.round(r.age_seconds / 60)} min old`}</span>`;
    const sc = r.scan;
    $('mkNote').textContent = sc.running ? `Signal scan running: ${sc.done} of ${sc.total} ${isStock ? 'largest stocks' : 'largest coins'} done. Others get a full signal when you open them.`
      : sc.finished_at ? `Signals cover the ${sc.total} largest ${isStock ? 'stocks' : 'coins'} (daily bars, refreshed every 3 hours). Open any market for its full analysis.`
      : 'The signal scan starts shortly after the server starts. Open any market for its full analysis now.';
    const sel = $('mkSector');
    sel.hidden = !isStock;
    if (isStock && sel.options.length <= 1) sel.innerHTML = '<option value="">All sectors</option>' + r.sectors.map(s => `<option>${esc(s)}</option>`).join('');
    const cols = [['rank', '#'], ['symbol', 'Market'], ['price', 'Price'], ['change', isStock ? 'Today' : '24h'], ['volume', 'Volume'],
      ['mcap', 'Market cap'], ['signal', 'Signal'], ['', 'RSI'], ['', '1 month']];
    $('mkTable').innerHTML = `<thead><tr>${cols.map(([k, n]) => `<th data-k="${k}" class="${k && k === mk.sort ? 'sorted' : ''}">${n}${k === mk.sort ? (mk.desc ? ' ↓' : ' ↑') : ''}</th>`).join('')}</tr></thead><tbody>` +
      r.rows.map(x => {
        const s = x.signal;
        return `<tr data-s="${esc(x.symbol)}" class="${x.analyzable ? '' : 'na'}" title="${x.analyzable ? 'Open full analysis' : 'Not listed on Binance: price only'}">
          <td>${x.rank ?? ''}</td><td><b>${esc(x.symbol)}</b><small>${esc(x.name)}${x.sector ? `, ${esc(x.sector)}` : ''}</small></td>
          <td>${money(x.price)}</td><td class="${(x.change_pct ?? 0) >= 0 ? 'up' : 'down'}">${x.change_pct == null ? '—' : `${x.change_pct >= 0 ? '+' : ''}${x.change_pct.toFixed(2)}%`}</td>
          <td>${big(x.volume)}</td><td>${big(x.mcap)}</td>
          <td>${s ? `<span class="pill ${lblTone(s.label)}">${esc(s.label)}</span>` : '<span style="color:var(--muted)">open to analyse</span>'}</td>
          <td>${s ? s.rsi.toFixed(0) : '—'}</td><td class="${s && s.ret_1m >= 0 ? 'up' : 'down'}">${s ? pct(s.ret_1m) : '—'}</td></tr>`;
      }).join('') + '</tbody>';
    const page = Math.floor(mk.offset / mk.limit) + 1, pages = Math.max(1, Math.ceil(r.matches / mk.limit));
    $('mkPage').textContent = `Page ${page} of ${pages}`;
    $('mkCount').textContent = `${r.matches.toLocaleString()} shown of ${r.total.toLocaleString()}`;
    $('mkPrev').disabled = page <= 1; $('mkNext').disabled = page >= pages;
  } catch (e) {
    $('marketsTag').textContent = 'unavailable';
    $('mkTable').innerHTML = `<tr><td class="empty">Market list could not be loaded: ${esc(e.message)}</td></tr>`;
  }
  mk.timer = setTimeout(() => loadMarkets(), state.asset === 'crypto' ? 30000 : 60000);  // keep prices live
}
let mkDeb;
$('mkSearch').oninput = e => { clearTimeout(mkDeb); mkDeb = setTimeout(() => { mk.q = e.target.value.trim(); loadMarkets(true); }, 250); };
$('mkSignal').onchange = e => { mk.signal = e.target.value; loadMarkets(true); };
$('mkSector').onchange = e => { mk.sector = e.target.value; loadMarkets(true); };
$('mkPrev').onclick = () => { mk.offset = Math.max(0, mk.offset - mk.limit); loadMarkets(); };
$('mkNext').onclick = () => { mk.offset += mk.limit; loadMarkets(); };
$('mkTable').onclick = e => {
  const th = e.target.closest('th[data-k]');
  if (th && th.dataset.k) {
    const k = th.dataset.k;
    mk.desc = mk.sort === k ? !mk.desc : ['change', 'volume', 'mcap', 'signal', 'price'].includes(k);
    mk.sort = k; return loadMarkets(true);
  }
  const tr = e.target.closest('tr[data-s]');
  if (tr && !tr.classList.contains('na')) { state.symbol = tr.dataset.s; state.interval = '1d'; load(); scrollTo({ top: 0, behavior: 'smooth' }); }
};

// ---------------------------------------------------------------- breaking news
let newsTopic = 'geopolitics', newsTimer = null;
async function loadFeed() {
  clearTimeout(newsTimer);
  try {
    const f = await api(`/api/news?topic=${newsTopic}`, 30000);
    $('newsTag').textContent = f.n ? `${f.n} stories, newest ${f.newest_age_min ?? '?'} min ago, from ${(f.sources_ok || []).join(', ') || 'no source'}` : 'no stories right now';
    $('feedThemes').innerHTML = (f.themes || []).map(t => `<span class="theme ${t.sentiment > 0.05 ? 'bull' : t.sentiment < -0.05 ? 'bear' : ''}">${esc(t.category)} ${t.n}</span>`).join('');
    $('feed').innerHTML = newsHtml(f.items || []);
  } catch (e) {
    $('newsTag').textContent = 'unavailable';
    $('feed').innerHTML = `<div class="empty">News could not be loaded: ${esc(e.message)}</div>`;
  }
  newsTimer = setTimeout(loadFeed, 180000);
}
$('newsTabs').onclick = e => {
  const t = e.target.dataset.topic; if (!t) return;
  newsTopic = t;
  document.querySelectorAll('#newsTabs button').forEach(b => b.classList.toggle('on', b.dataset.topic === t));
  loadFeed();
};

// ---------------------------------------------------------------- universe scan + ticker
async function loadScan() {
  try {
    const rep = await api('/api/report');
    $('scanTag').textContent = `generated ${rep.generated}`;
    const cols = [['Market', r => `${r.symbol} <small style="color:var(--muted)">${r.interval}, h${r.horizon}</small>`], ['Acc.', r => pct(r.accuracy)],
      ['Base', r => pct(r.baseline)], ['AUC', r => num(r.auc, 3)], ['p', r => num(r.p_value, 3)],
      ['Sharpe', r => `<span class="${r.sharpe > r.bh_sharpe ? 'up' : ''}">${num(r.sharpe)}</span>`], ['B&H', r => num(r.bh_sharpe)],
      ['Model', r => num(r.ml_sharpe)], ['Max DD', r => pct(r.max_dd, 0)], ['B&H DD', r => pct(r.bh_max_dd, 0)],
      ['Signal', r => `<span class="${/LONG|BUY/.test(r.signal) ? 'up' : 'amber'}">${esc(r.signal)}</span>`]];
    const rows = rep.rows.slice().sort((a, b) => b.sharpe - a.sharpe);
    $('scan').innerHTML = `<thead><tr>${cols.map(c => `<th>${c[0]}</th>`).join('')}</tr></thead><tbody>` +
      rows.map(r => `<tr data-s="${esc(r.symbol)}" data-a="${esc(r.asset)}" data-i="${esc(r.interval)}" data-h="${r.horizon}">${cols.map(c => `<td>${c[1](r)}</td>`).join('')}</tr>`).join('') + '</tbody>';
  } catch { $('scan').innerHTML = '<tr><td class="empty">Run <code>python3 scripts/evaluate.py</code> to fill the universe scan.</td></tr>'; }
}

async function loadTicker() {
  const syms = state.universe[state.asset];
  const out = await Promise.all(syms.map(s => api(`/api/quote?symbol=${s}&asset=${state.asset}`).then(q => [s, q]).catch(() => null)));
  const html = out.filter(Boolean).map(([s, q]) =>
    `<span data-s="${s}">${s} <b>${money(q.price)}</b> <em class="${q.change_pct >= 0 ? 'up' : 'down'}" style="font-style:normal">${q.change_pct >= 0 ? '▲' : '▼'}${Math.abs(q.change_pct).toFixed(2)}%</em></span>`).join('');
  $('ticker').innerHTML = html ? `<div class="track">${html}${html}</div>` : '';
}

// ---------------------------------------------------------------- views (one job per screen)
const VIEWS = ['overview', 'timeframes', 'why', 'economy', 'markets', 'news', 'record', 'lab'];
function setView(v, push = true) {
  if (!VIEWS.includes(v)) v = 'overview';
  document.querySelectorAll('#views button').forEach(b => b.classList.toggle('on', b.dataset.view === v));
  document.querySelectorAll('main [data-view]').forEach(s => s.classList.toggle('show', s.dataset.view === v));
  if (push && location.hash !== `#${v}`) history.replaceState(null, '', `#${v}`);
  // Canvases can't be measured while hidden: redraw charts once their view is visible.
  if ((v === 'overview' || v === 'lab') && state.last) requestAnimationFrame(() => render(state.last));
  scene?.setView?.(v);
}
$('views').onclick = e => { const v = e.target.dataset.view; if (v) setView(v); };
document.addEventListener('click', e => {
  const g = e.target.closest('[data-goto]');
  if (g) { setView(g.dataset.goto); scrollTo({ top: $('views').offsetTop - 10, behavior: 'smooth' }); }
});
addEventListener('hashchange', () => setView(location.hash.slice(1), false));
setView(location.hash.slice(1) || 'overview', false);

// ---------------------------------------------------------------- events
$('assetSeg').onclick = e => {
  const a = e.target.dataset.asset; if (!a || a === state.asset) return;
  state.asset = a; state.symbol = state.universe[a][0]; state.interval = '1d';
  load(); loadTicker(); loadMarkets(true);
};
$('symbols').onclick = e => { const s = e.target.dataset.s; if (s) { state.symbol = s; load(); } };
$('ticker').onclick = e => { const s = e.target.closest('span')?.dataset.s; if (s) { state.symbol = s; load(); } };
$('interval').onchange = e => { state.interval = e.target.value; load(); };
$('horizon').onchange = e => { state.horizon = +e.target.value; load(); };
$('custom').onsubmit = e => {
  e.preventDefault();
  const s = $('customSym').value.trim().toUpperCase().replace(/[^A-Z0-9.\-]/g, '');
  if (!s) return;
  // A known ticker from the other asset class switches tabs automatically.
  const other = state.asset === 'crypto' ? 'stock' : 'crypto';
  if (!state.universe.all?.[state.asset]?.includes(s) && state.universe.all?.[other]?.includes(s)) {
    state.asset = other; state.interval = '1d'; loadTicker();
  }
  state.symbol = s; $('customSym').value = ''; load();
};
$('customSym').oninput = e => {  // picking from the autocomplete list analyses immediately
  const v = e.target.value.toUpperCase();
  if (state.universe.all?.[state.asset]?.includes(v) && e.inputType === 'insertReplacementText') $('custom').requestSubmit();
};
$('scan').onclick = e => {
  const tr = e.target.closest('tr[data-s]'); if (!tr) return;
  Object.assign(state, { symbol: tr.dataset.s, asset: tr.dataset.a, interval: tr.dataset.i, horizon: +tr.dataset.h });
  load(); scrollTo({ top: 0, behavior: 'smooth' });
};
let rz; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => state.last && render(state.last), 200); });

// ---------------------------------------------------------------- start
(async () => {
  intro.step(0.1, 'Connecting to the market data feeds');
  try {
    state.universe = await api('/api/universe');
  } catch (e) {
    intro.step(1, 'Cannot reach the server');
    banner('Cannot reach the server. Start it with <code>python3 server.py</code> and reload this page.');
    intro.finish();
    return;
  }
  intro.step(0.25, 'Loading the 3D engine');
  await Promise.race([sceneReady, new Promise(r => setTimeout(r, 1500))]);
  scene?.playIntro?.(reduceMotion ? 0 : 2.6);
  setTimeout(() => intro.finish(), 6000);  // never hold the page hostage to a slow data source
  checkServer();
  load(); loadScan(); loadTicker(); loadMarkets(true); loadFeed();
  setInterval(loadTicker, 30000);
  setInterval(checkServer, 60000);
})();
