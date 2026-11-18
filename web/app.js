import { Scene3D } from './scene3d.js?v=11';
import { candleChart, equityChart, calibrationChart } from './charts.js?v=11';

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

async function api(path, timeoutMs = 90000, body = null) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  // Actions that change data are POSTs carrying this session's CSRF token.
  const opts = body === null ? { signal: ctl.signal } : {
    method: 'POST', signal: ctl.signal, body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': state.csrf || '' } };
  let r;
  try { r = await fetch(path, opts); }
  catch (e) { throw new Error(e.name === 'AbortError' ? `The server took longer than ${timeoutMs / 1000}s to answer. Data sources may be slow; try again.` : 'Cannot reach the server.'); }
  finally { clearTimeout(timer); }
  let js;
  try { js = await r.json(); } catch { throw new Error(`Server returned ${r.status} ${r.statusText}`); }
  if (r.status === 401 && js.signin) { location.replace('/login'); throw new Error('Signed out.'); }
  if (!r.ok) throw new Error(js.error || r.statusText);
  return js;
}
const apiPost = (path, body, timeoutMs = 30000) => api(path, timeoutMs, body || {});

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

// The server reloads itself when its code changes. The page remembers the code version it
// was loaded with and offers a one-click reload when a newer one is running.
let loadedVersion = null;
async function checkServer() {
  let v;
  try {
    v = await api('/api/version', 10000);
  } catch {
    return banner('Cannot reach the server. If it stopped, start it again with <code>python3 server.py</code>; this page reconnects by itself.');
  }
  if (!v.py) {  // an old server without self-reload is still running
    return banner('An older version of the server is still running. Restart it once: <code>lsof -ti tcp:8000 | xargs kill; python3 server.py</code>. After that it updates itself.');
  }
  loadedVersion = loadedVersion || v;
  if (v.reload_error) {
    banner(`The latest code change has an error, so the server is still running the previous version: <code>${esc(v.reload_error)}</code>`);
  } else if (v.py !== loadedVersion.py || v.web !== loadedVersion.web) {
    banner('The app was updated. <button class="banner-btn" id="reloadBtn">Reload to get the new version</button>');
    $('reloadBtn').onclick = () => location.reload();
  } else if (state.universe?.demo) {
    banner('Demo mode: every price on this page is synthetic. Start the server without <code>QUANT_DEMO=1</code> for real markets.');
  } else {
    banner('');
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
  // The newest closed candle can legitimately be up to one candle old (a daily candle closes once a day),
  // so judge freshness by missing candles, not by age.
  const fr = d.freshness || { fresh: (d.data_age_seconds ?? 0) < 3600, missing_bars: 0 };
  const age = fr.fresh ? 0 : 99999;
  const fresh = fr.fresh ? 'newest closed candle' : `${fr.missing_bars || 1} candle${fr.missing_bars > 1 ? 's' : ''} behind`;
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
  // Default view: only calls made under the current rules, the number that says whether they work.
  const rulesView = state.learnScope !== 'all' && j.current_rules;
  const a = rulesView ? j.current_rules : j.asset_class, mk = j.market;
  const sinceTxt = j.rules_since ? new Date(j.rules_since * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : null;
  document.querySelectorAll('#learnScope button').forEach(b => b.classList.toggle('on', (b.dataset.scope === 'rules') === !!rulesView));
  $('learnScope').hidden = !j.current_rules;
  const cls = d.asset_class === 'crypto' ? 'crypto' : 'stock';
  $('learnTag').textContent = `${rulesView ? `since ${sinceTxt}: ` : ''}${a.total.toLocaleString()} ${cls} calls logged, ${a.resolved.toLocaleString()} checked, ${a.open.toLocaleString()} waiting`;
  $('learnSummary').textContent = a.resolved === 0
    ? `Learning has started. Every signal the bot gives is written to its journal with the price at that moment, then checked automatically when it comes due (next bar, 5 bars, 10 bars, by timeframe). Hit rates appear here as soon as the first calls resolve, and factor weights start adapting after ${40} checked swing calls.`
    : `${a.learning_active ? 'Learning is active: factor weights are being adjusted from' : 'Collecting evidence:'} ${a.resolved.toLocaleString()} checked ${cls} calls. ` +
      `For ${d.symbol}: ${mk.resolved} checked, ${mk.open} waiting. Hit rates below are live, out-of-sample results, not backtests.`;
  if (rulesView) $('learnSummary').textContent = `Showing only calls made since the signal fix on ${sinceTxt}, ` +
    `when 15m/4h stopped making calls and 30m/1h switched to reversal-only. ${a.resolved ? `${a.resolved} of them have been checked so far.` : 'None has come due yet.'} ` +
    'Switch to "All calls" to include the older rules.';
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

$('learnScope').onclick = e => {
  const s = e.target.dataset.scope; if (!s) return;
  state.learnScope = s;
  if (state.last) renderLearning(state.last);
};

// ---------------------------------------------------------------- research desk (20 agents)
const GATE_TONE = { GO: 'bull', REVIEW_REQUIRED: 'neutral', NO_GO: 'bear', NO_ACTIONABLE_ORDERS: 'unknown' };
const GATE_LABEL = { GO: 'GO', REVIEW_REQUIRED: 'Review first', NO_GO: 'Blocked', NO_ACTIONABLE_ORDERS: 'Watchlist' };
let deskTimer = null, deskSeen = null;
const fmtv = v => typeof v === 'number' ? (Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : Number(v.toPrecision(6)).toString()) : esc(v);

async function loadDesk() {
  clearTimeout(deskTimer);
  let d;
  try { d = await api('/api/agents', 15000); } catch (e) { $('deskTag').textContent = 'unavailable'; return; }
  const st = d.state;
  $('roster').innerHTML = d.roster.map(a => `<div class="agent"><b>${esc(a.name)}</b><span class="chip neutral">${esc(a.skills[0] || '')}</span>
      <p>${esc(a.job)}</p>${a.needs ? `<small>${esc(a.needs)}</small>` : ''}${st.timings[a.name] != null ? `<small>last run ${st.timings[a.name]}s</small>` : ''}</div>`).join('');
  $('pipeline').innerHTML = st.running
    ? `<span class="ring inline"></span> Running: <b>${esc(st.stage)}</b> (${Math.round(Date.now() / 1000 - st.started)}s)`
    : st.error ? `<span class="err-box"><b>The last run failed:</b> ${esc(st.error)}</span>` : '';
  if (d.result && d.result.finished !== deskSeen) { deskSeen = d.result.finished; renderAgentDesk(d.result); }
  renderAlerts(d.result, d.rung);
  if (st.running) deskTimer = setTimeout(loadDesk, 2000);
}

function signalCard(s) {
  const g = s.gate, v = s.values, tone = GATE_TONE[g.decision] || 'neutral';
  const rp = s.context.replay;
  const ticket = (state.deskTickets || []).find(t => t.signal_id === s.id);
  const levels = s.setup === 'pair_spread'
    ? `spread z ${fmtv(v.z)} &rarr; target 0, stop ${fmtv(v.stop)}`
    : `entry ${fmtv(v.entry)} &middot; stop ${fmtv(v.stop)} &middot; target ${fmtv(v.target_1)}`;
  return `<details class="gsig ${tone}"><summary>
      <span class="chip ${tone}">${GATE_LABEL[g.decision] || g.decision}</span>
      <b>${esc(s.symbol)}</b><span class="${s.side === 'LONG' ? 'up' : 'down'}">${s.side}</span>
      <span class="gsetup">${esc(s.setup.replace(/_/g, ' '))}, ${esc(s.timeframe)}</span>
      <span class="glevels">${levels}</span>
      ${v.shares ? `<span class="gsize">${fmtv(v.shares)} ${s.asset === 'stock' ? 'sh' : 'units'}, risk $${fmtv(v.planned_risk)}</span>` : ''}
      ${rp ? `<span class="godds">target first ${pct(rp.p_target, 0)}, ${rp.expected_r >= 0 ? '+' : ''}${rp.expected_r.toFixed(2)}R</span>` : ''}
    </summary>
    <div class="gbody">
      <div><h4>Gate decision</h4><ul>${g.reasons.map(r => `<li>${esc(r)}</li>`).join('')}</ul>
        ${ticket ? `<h4>Order ticket (you place it)</h4><p class="ticket">${esc(ticket.text)}</p>
          ${ticket.order ? `<button class="link-btn copy" data-copy="${esc(JSON.stringify(ticket.order))}">Copy order JSON</button>` : ''}` : ''}</div>
      <div><h4>How the agents got here</h4><ol class="trail">${s.notes.map(n => `<li><b>${esc(n.agent)}</b> ${esc(n.text)}</li>`).join('')}</ol>
        <details class="more"><summary>Trace every number</summary><table class="stats">${(s.context.trace || []).map(t =>
          `<tr><td>${esc(t.name)}</td><td>${fmtv(t.value)}</td><td>${esc(t.source)}</td><td><code>${esc(t.code)}</code></td></tr>`).join('')}</table></details></div>
    </div></details>`;
}

function renderAgentDesk(r) {
  state.deskTickets = r.tickets || [];
  const c = r.counts || {};
  $('gateCounts').textContent = `${r.asset}: ${c.GO || 0} GO, ${c.REVIEW_REQUIRED || 0} review, ${c.NO_GO || 0} blocked, ${c.NO_ACTIONABLE_ORDERS || 0} watchlist` +
    (r.withheld_ungated ? `, ${r.withheld_ungated} withheld (never gated)` : '') + `, ${new Date(r.finished * 1000).toLocaleTimeString()}`;
  $('gSignals').innerHTML = r.signals.length ? r.signals.map(signalCard).join('')
    : `<div class="empty">No setups passed the agents this run${r.scan ? ` (${r.scan.scanned} markets scanned, ${r.scan.hits.length} chart setups found)` : ''}. That is a normal answer: no trade is a position too.</div>`;
  const th = r.theses || [];
  $('thesesTable').innerHTML = th.length ? `<thead><tr><th>Thesis</th><th>Status</th><th>Plan</th><th>Result</th><th></th></tr></thead><tbody>` +
    th.slice().reverse().map(t => `<tr><td><b>${esc(t.symbol)}</b> <small style="color:var(--muted)">${esc(t.setup)}, ${esc(t.side)}</small></td>
      <td>${esc(t.status)}</td><td>entry ${fmtv(t.plan.entry)}, stop ${fmtv(t.plan.stop)}</td>
      <td>${t.status === 'CLOSED' ? `<span class="${t.pnl >= 0 ? 'up' : 'down'}">$${fmtv(t.pnl)} (${t.r_multiple >= 0 ? '+' : ''}${t.r_multiple}R)</span>` : ''}</td>
      <td>${t.status === 'ENTRY_READY' || t.status === 'IDEA' ? `<button class="link-btn th" data-act="open" data-id="${esc(t.id)}">I bought it</button>`
        : t.status === 'ACTIVE' ? `<button class="link-btn th" data-act="close" data-id="${esc(t.id)}">I closed it</button>` : ''}</td></tr>`).join('') + '</tbody>'
    : '<tr><td class="empty">No theses yet. Every gated idea is recorded here; mark the ones you actually trade.</td></tr>';
  const coach = r.coach || {};
  $('learnBox').innerHTML = `<h4>Coach</h4><p class="lead small">${esc(coach.summary || '')}</p>
      ${(coach.findings || []).map(f => `<div class="tk neutral"><i>●</i><div>${esc(f)}</div></div>`).join('')}
      ${(coach.rules || []).length ? `<h4>Next-session rules</h4><ul>${coach.rules.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}
    <h4>Research hypotheses</h4>${(r.research || []).map(h => `<details class="fx neutral"><summary><span class="fx-name">${esc(h.verdict)}</span><span></span>
      <span class="fx-head">${esc(h.hypothesis)}</span><span class="chip neutral">test</span></summary><p>${esc(h.evidence)} Test: ${esc(h.test)} Kill: ${esc(h.kill)}</p></details>`).join('')}
    <h4>Pivots</h4>${(r.pivots || []).map(p => `<p class="lead small">${esc(p.strategy)}: ${esc(p.evidence)}${p.pivots.length ? ` ${esc(p.pivots.join(' '))}` : ''}</p>`).join('')}`;
  const parts = [];
  if (r.pairs?.length) parts.push(`<h4>Pairs (most cointegrated first)</h4><table class="scan"><tr><th>Pair</th><th>Correlation</th><th>ADF t</th><th>Spread z</th><th>Half-life</th></tr>` +
    r.pairs.slice(0, 10).map(p => `<tr><td>${esc(p.pair)} <small style="color:var(--muted)">${esc(p.group)}</small></td><td>${p.corr.toFixed(2)}</td>
      <td class="${p.cointegrated ? 'up' : ''}">${p.adf_t.toFixed(2)}</td><td>${p.z >= 0 ? '+' : ''}${p.z.toFixed(2)}</td><td>${isFinite(p.half_life) ? p.half_life.toFixed(0) + 'd' : '—'}</td></tr>`).join('') + '</table>');
  if (r.earnings?.length) parts.push(`<h4>Recent earnings reactions</h4><table class="scan"><tr><th>Stock</th><th>Grade</th><th>Score</th><th>Gap</th><th>Volume</th></tr>` +
    r.earnings.slice(0, 12).map(e => `<tr><td>${esc(e.symbol)} <small style="color:var(--muted)">${esc(e.report_date)}</small></td><td>${esc(e.grade)}</td>
      <td>${e.score.toFixed(0)}</td><td class="${e.gap >= 0 ? 'up' : 'down'}">${pct(e.gap)}</td><td>${e.volume_ratio.toFixed(1)}x</td></tr>`).join('') + '</table>');
  if (r.congress?.filings) parts.push(`<h4>Congress trade filings</h4><p class="lead small">${esc(r.congress.note)}</p>` +
    r.congress.filings.slice(0, 10).map(f => `<a class="cg" href="${esc(safeLink(f.link))}" target="_blank" rel="noopener noreferrer">
      ${f.new ? '<span class="chip bull">new</span>' : ''} ${esc(f.member)} (${esc(f.district || '')}), filed ${esc(f.filed)}
      ${f.tickers.length ? `: ${esc(f.tickers.join(', '))}` : ''}${f.watched.length ? ` <b class="amber">matches: ${esc(f.watched.join(', '))}</b>` : ''}</a>`).join(''));
  $('ctxBox').innerHTML = parts.join('') || '<div class="empty">Pairs, earnings and Congress results appear here after a run (earnings and Congress on the Stocks desk).</div>';
  $('ctxTag').textContent = r.asset === 'stock' ? 'stock desk' : 'crypto desk';
}

function renderAlerts(r, rung) {
  const pending = (r && r.alerts) || [];
  $('alertsList').innerHTML = (rung.length ? rung.slice().reverse().map(a => `<div class="tk bear"><i>!</i><div><b>${esc(a.symbol)}</b> hit its ${esc(a.kind)} (${fmtv(a.level)}) at ${fmtv(a.rang.price)}<small>${esc(a.rang.at)}</small></div></div>`).join('') : '')
    + (pending.length ? pending.map(a => `<div class="tk neutral"><i>●</i><div>${esc(a.symbol)} ${esc(a.kind)}: rings when price goes ${esc(a.when)} ${fmtv(a.level)}</div></div>`).join('')
       : '<div class="empty">No alerts. GO signals get stop and target alerts automatically; alerts inform, they never trade.</div>');
}

$('askForm').onsubmit = async e => {
  e.preventDefault();
  const q = $('askInput').value.trim();
  if (q) {
    try { const r = await api(`/api/agents/navigate?q=${encodeURIComponent(q)}`); $('askAnswer').textContent = r.answer; } catch { /* routing is optional */ }
  }
  const r = await apiPost('/api/agents/run', { asset: state.asset, size: $('deskSize').value }).catch(err => ({ note: err.message }));
  if (r.note) $('askAnswer').textContent = r.note;
  loadDesk();
};
document.addEventListener('click', async e => {
  const cp = e.target.closest('.copy');
  if (cp) { navigator.clipboard?.writeText(cp.dataset.copy); cp.textContent = 'Copied'; return; }
  const b = e.target.closest('.th');
  if (!b) return;
  const opening = b.dataset.act === 'open';
  const price = prompt(opening ? 'Fill price you got:' : 'Exit price you got:');
  if (!price) return;
  const qty = opening ? prompt('Quantity filled:') : '';
  if (opening && !qty) return;
  try {
    await apiPost('/api/agents/thesis', { action: b.dataset.act, id: b.dataset.id, price, qty });
    deskSeen = null; loadDesk();
  } catch (err) { alert(err.message); }
});

// ---------------------------------------------------------------- views (one job per screen)
const VIEWS = ['overview', 'timeframes', 'why', 'economy', 'markets', 'news', 'record', 'lab', 'agents', 'paper', 'account', 'team'];
function setView(v, push = true) {
  if (!VIEWS.includes(v)) v = 'overview';
  document.querySelectorAll('#views button').forEach(b => b.classList.toggle('on', b.dataset.view === v));
  document.querySelectorAll('main [data-view]').forEach(s => s.classList.toggle('show', s.dataset.view === v));
  if (push && location.hash !== `#${v}`) history.replaceState(null, '', `#${v}`);
  // Canvases can't be measured while hidden: redraw charts once their view is visible.
  if ((v === 'overview' || v === 'lab') && state.last) requestAnimationFrame(() => render(state.last));
  if (v === 'agents') loadDesk();
  if (v === 'paper') loadPaper();
  if (v === 'account') loadAccount();
  if (v === 'team') loadTeam();
  scene?.setView?.(v);
}
$('views').onclick = e => { const v = e.target.dataset.view; if (v) setView(v); };
document.addEventListener('click', e => {
  const g = e.target.closest('[data-goto]');
  if (g) { setView(g.dataset.goto); scrollTo({ top: $('views').offsetTop - 10, behavior: 'smooth' }); }
});
addEventListener('hashchange', () => setView(location.hash.slice(1), false));
setView(location.hash.slice(1) || 'overview', false);

// ---------------------------------------------------------------- demo trades (paper trading)
const rr = v => v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}R`;
const usd = v => v == null ? '—' : `${v < 0 ? '−' : ''}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
const DEC = { GO: 'GO', REVIEW_REQUIRED: 'Review first', NO_GO: 'Blocked' };
const VERDICT = { edge: ['up', 'real edge'], failing: ['down', 'losing: gate blocks it'], losing: ['amber', 'losing: gate reviews it'],
  unproven: ['amber', 'no clear edge yet'], 'too few': ['', 'collecting trades'] };
let paperTimer = null;

async function loadPaper() {
  clearTimeout(paperTimer);
  let p;
  try { p = await api('/api/paper', 20000); }
  catch (err) { $('paperKpis').innerHTML = `<p class="err">${esc(err.message)}</p>`; return; }
  const a = p.account;
  $('paperAuto').checked = !!p.auto;
  $('paperTag').textContent = p.desk.running ? `desk running: ${p.desk.stage}` :
    p.auto ? `autopilot on, every ${p.every_minutes} min${p.last_run ? `; last ${new Date(p.last_run.at * 1000).toLocaleTimeString()} (${p.last_run.asset})` : ''}` : 'autopilot off';
  const ret = (a.balance - a.start) / a.start;
  const kpi = (label, val, cls = '', sub = '') => `<div class="kpi"><small>${label}</small><b class="${cls}">${val}</b>${sub ? `<i>${sub}</i>` : ''}</div>`;
  $('paperKpis').innerHTML = [
    kpi('Balance', usd(a.balance), ret >= 0 ? 'up' : 'down', pct(ret, 2)),
    kpi('Open P&L', usd(a.unrealised), a.unrealised >= 0 ? 'up' : 'down', `${a.open} open`),
    kpi('Closed trades', a.n || 0),
    kpi('Winners', a.n ? pct(a.win_rate, 0) : '—'),
    kpi('Average trade', a.n ? rr(a.avg_r) : '—', a.avg_r > 0 ? 'up' : a.avg_r < 0 ? 'down' : '', 'R = what was risked'),
    kpi('Profit factor', a.profit_factor == null ? '—' : num(a.profit_factor), '', 'gains ÷ losses'),
    kpi('Worst drawdown', a.n ? pct(a.max_drawdown, 1) : '—'),
    kpi('Stops / targets', a.n ? `${a.stops} / ${a.targets}` : '—'),
  ].join('');
  $('paperEquity').hidden = p.equity.length < 2;
  if (p.equity.length >= 2) equityChart($('paperEquity'), [{ t: p.equity[0].t - 1, s: 1, b: 1 }, ...p.equity]);

  // Promised vs delivered: the replay's predicted odds against what the demo trades actually did.
  const c = p.calibration, all = [p.account, p.shadow].filter(x => x.n);
  const n = all.reduce((t, x) => t + x.n, 0);
  $('paperTruth').innerHTML = !c.n ? '<p class="lead small">No closed demo trades yet. Trades close at their stop, target or time limit; for daily setups that can take days. Hourly crypto setups close within about 6 hours.</p>' :
    `<p class="lead small">Over ${c.n} closed demo trades (both books), before any trade the desk predicted:</p>
    <div class="kpis">
      ${kpi('Target hit: predicted', pct(c.predicted_target, 0))}${kpi('Target hit: actual', pct(c.actual_target, 0), Math.abs(c.actual_target - c.predicted_target) < 0.08 ? 'up' : 'amber')}
      ${kpi('Average trade: predicted', rr(c.predicted_r))}${kpi('Average trade: actual', rr(c.actual_r), c.actual_r > 0 ? 'up' : 'down', 'after fees and slippage')}
    </div>
    <p class="lead small">${c.actual_r > 0 && c.actual_r >= c.predicted_r - 0.1 ? 'So far the signals are delivering roughly what they promise.'
      : c.actual_r > 0 ? 'Profitable, but below what the replay promised: treat its numbers as optimistic.'
      : 'So far the signals are <b>not</b> making money after costs. The gate tightens automatically as setups prove themselves losers.'}
      ${n < 30 ? ' This is still a small sample: under 30 trades, luck dominates.' : ''}</p>`;

  $('paperGate').textContent = p.gate_verdict;
  $('paperDecisions').innerHTML = `<thead><tr><th>Gate said</th><th>Closed</th><th>Winners</th><th>Average</th><th>Total</th></tr></thead><tbody>${
    Object.entries(p.by_decision).map(([d, s]) => `<tr><td>${DEC[d]}</td><td>${s.n}</td><td>${s.n ? pct(s.win_rate, 0) : '—'}</td>
      <td class="${s.avg_r > 0 ? 'up' : s.avg_r < 0 ? 'down' : ''}">${s.n ? rr(s.avg_r) : '—'}</td><td>${s.n ? rr(s.total_r) : '—'}</td></tr>`).join('')}</tbody>`;

  const cell = s => !s.n ? '<td>—</td><td>—</td><td>—</td>'
    : `<td>${s.n}</td><td>${pct(s.win_rate, 0)}</td><td class="${s.avg_r > 0 ? 'up' : s.avg_r < 0 ? 'down' : ''}">${rr(s.avg_r)}</td>`;
  $('paperStyles').innerHTML = p.by_style.length
    ? `<thead><tr><th>Style</th><th>Open now</th><th>Long: closed</th><th>Winners</th><th>Average</th><th>Short: closed</th><th>Winners</th><th>Average</th></tr></thead><tbody>${
      p.by_style.map(s => `<tr><td><b>${esc(s.style)}</b></td><td>${s.open}</td>${cell(s.long)}${cell(s.short)}</tr>`).join('')}</tbody>`
    : '<tr><td class="lead small">No demo trades yet.</td></tr>';

  $('paperSetups').innerHTML = p.by_setup.length ? `<thead><tr><th>Setup</th><th>Market</th><th>Closed</th><th>Winners</th><th>Average</th><th>Confidence</th><th>Verdict</th></tr></thead><tbody>${
    p.by_setup.map(s => { const [cls, txt] = VERDICT[s.verdict] || ['', s.verdict];
      return `<tr><td>${esc(s.setup.replace(/_/g, ' '))}</td><td>${esc(s.asset)}</td><td>${s.n}${s.n < s.need ? ` <small>of ${s.need}</small>` : ''}</td>
        <td>${s.n ? pct(s.win_rate, 0) : '—'}</td><td class="${s.avg_r > 0 ? 'up' : s.avg_r < 0 ? 'down' : ''}">${s.n ? rr(s.avg_r) : '—'}</td>
        <td>${s.t == null ? '—' : `t = ${s.t.toFixed(1)}`}</td><td><span class="tag ${cls}">${txt}</span></td></tr>`; }).join('')}</tbody>`
    : '<tr><td class="lead small">No demo trades yet. Run the desk from the Agents tab, or leave autopilot on.</td></tr>';

  $('paperOpenTag').textContent = `${p.open.length} open or waiting`;
  $('paperOpen').innerHTML = p.open.length ? `<thead><tr><th>Market</th><th>Setup</th><th>Style</th><th>Side</th><th>Book</th><th>Status</th><th>Entry</th><th>Stop</th><th>Target</th><th>Now</th><th>Open P&L</th><th>Closes by</th></tr></thead><tbody>${
    p.open.map(t => `<tr><td><b>${esc(t.symbol)}</b> <small>${esc(t.asset)}</small></td><td>${esc(t.setup.replace(/_/g, ' '))}</td>
      <td>${esc(t.style)}</td><td class="${t.side === 'LONG' ? 'up' : 'down'}">${esc(t.side)}</td>
      <td>${t.book === 'account' ? 'account' : `shadow <small>(${DEC[t.decision]})</small>`}</td>
      <td>${t.status === 'OPEN' ? 'open' : `waiting: ${t.order_type === 'stop' ? 'breakout trigger' : t.order_type === 'open' ? 'market open' : 'price'}`}</td>
      <td>${money(t.fill ?? t.entry_plan)}</td><td>${t.setup === 'pair_spread' ? `z ${num(t.stop)}` : money(t.stop)}</td>
      <td>${t.setup === 'pair_spread' ? 'z 0' : money(t.target)}</td><td>${t.setup === 'pair_spread' ? (t.last_price != null ? `z ${num(t.last_price)}` : '—') : money(t.last_price)}</td>
      <td class="${t.open_r > 0 ? 'up' : t.open_r < 0 ? 'down' : ''}">${t.open_r == null ? '—' : rr(t.open_r)}</td>
      <td>${t.until ? when(t.until) : '—'}</td></tr>`).join('')}</tbody>`
    : '<tr><td class="lead small">No open demo trades.</td></tr>';

  $('paperHistory').innerHTML = p.closed.length ? `<thead><tr><th>Closed</th><th>Market</th><th>Setup</th><th>Style</th><th>Side</th><th>Book</th><th>Gate</th><th>Entry</th><th>Exit</th><th>Why</th><th>Result</th><th>P&L</th><th>Best / worst</th></tr></thead><tbody>${
    p.closed.map(t => t.status === 'CANCELLED' || t.status === 'VOID'
      ? `<tr class="muted"><td>${when(t.closed)}</td><td><b>${esc(t.symbol)}</b></td><td>${esc(t.setup.replace(/_/g, ' '))}</td><td>${esc(t.style)}</td><td>${esc(t.side)}</td><td>${esc(t.book)}</td><td>${DEC[t.decision] || ''}</td><td colspan="6">${t.status === 'VOID' ? 'voided, not counted' : 'not filled'}: ${esc(t.exit_reason)}</td></tr>`
      : `<tr><td>${when(t.closed)}</td><td><b>${esc(t.symbol)}</b> <small>${esc(t.asset)}</small></td><td>${esc(t.setup.replace(/_/g, ' '))}</td>
      <td>${esc(t.style)}</td><td class="${t.side === 'LONG' ? 'up' : 'down'}">${esc(t.side)}</td>
      <td>${esc(t.book)}</td><td>${DEC[t.decision] || ''}</td><td>${money(t.fill)}</td><td>${money(t.exit)}</td><td>${esc(t.exit_reason)}</td>
      <td class="${t.r > 0 ? 'up' : 'down'}">${rr(t.r)}</td><td class="${t.pnl > 0 ? 'up' : 'down'}">${usd(t.pnl)}</td>
      <td>${t.mfe_r == null ? '—' : `${rr(t.mfe_r)} / ${rr(t.mae_r)}`}</td></tr>`).join('')}</tbody>`
    : '<tr><td class="lead small">No closed demo trades yet.</td></tr>';

  if ($('views').querySelector('button.on')?.dataset.view === 'paper') paperTimer = setTimeout(loadPaper, 30000);
}
$('paperAuto').onchange = async e => {
  try { await apiPost('/api/paper/auto', { on: e.target.checked }); } catch (err) { alert(err.message); }
  loadPaper();
};
$('paperCheck').onclick = async () => {
  $('paperCheck').disabled = true;
  try { const r = await apiPost('/api/paper/check', {}, 120000); $('paperCheck').textContent = r.closed ? `${r.closed} closed` : 'Checked'; }
  catch (err) { alert(err.message); }
  $('paperCheck').disabled = false;
  setTimeout(() => { $('paperCheck').textContent = 'Check open trades now'; }, 3000);
  loadPaper();
};

// ---------------------------------------------------------------- account & team
const when = t => t ? new Date(t * 1000).toLocaleString() : '—';

function renderUser() {
  const u = state.user;
  $('whoami').textContent = u ? `${u.name || u.email}${u.role === 'admin' ? ' · admin' : ''}` : '';
  $('teamTab').hidden = u?.role !== 'admin';
}
$('signOut').onclick = async () => {
  try { await apiPost('/api/auth/logout'); } catch { /* signed out either way */ }
  location.replace('/login');
};

function loadAccount() {
  const on = !!state.user?.two_factor;
  $('tfaTag').textContent = on ? 'on' : 'off';
  $('tfaTag').className = `tag ${on ? 'up' : 'amber'}`;
  $('tfaText').textContent = on
    ? 'Signing in needs your password and a 6-digit code from your authenticator app.'
    : 'Add a 6-digit code from an authenticator app (Google Authenticator, 1Password, Authy) on top of your password. Recommended for admins.';
  $('tfaBox').innerHTML = on
    ? `<form class="inline-form" id="tfaOff"><input id="tfaPw" type="password" placeholder="Your password" autocomplete="current-password" aria-label="Password"><button>Turn off</button></form><p class="lead small" id="tfaMsg"></p>`
    : `<button id="tfaStart" type="button">Set up two-step sign-in</button><p class="lead small" id="tfaMsg"></p>`;
  if (on) {
    $('tfaOff').onsubmit = async e => {
      e.preventDefault();
      try { await apiPost('/api/auth/2fa/disable', { password: $('tfaPw').value }); state.user.two_factor = false; loadAccount(); }
      catch (err) { $('tfaMsg').textContent = err.message; }
    };
  } else {
    $('tfaStart').onclick = async () => {
      let s;
      try { s = await apiPost('/api/auth/2fa/begin'); } catch (err) { $('tfaMsg').textContent = err.message; return; }
      $('tfaBox').innerHTML = `
        <p class="lead small">In your authenticator app choose “enter a setup key” and type this key, or paste the link into an app that accepts it.</p>
        <div class="linkbox"><code>${esc(s.secret.replace(/(.{4})/g, '$1 ').trim())}</code></div>
        <div class="linkbox"><code>${esc(s.uri)}</code></div>
        <form class="inline-form" id="tfaOn"><input id="tfaCode" inputmode="numeric" maxlength="6" placeholder="6-digit code" autocomplete="one-time-code" aria-label="Code"><button>Confirm and turn on</button></form>
        <p class="lead small" id="tfaMsg"></p>`;
      $('tfaOn').onsubmit = async e => {
        e.preventDefault();
        try { await apiPost('/api/auth/2fa/enable', { secret: s.secret, code: $('tfaCode').value }); state.user.two_factor = true; loadAccount(); }
        catch (err) { $('tfaMsg').textContent = err.message; }
      };
    };
  }
}
$('pwForm').onsubmit = async e => {
  e.preventDefault();
  try {
    const r = await apiPost('/api/auth/password', { old: $('pwOld').value, new: $('pwNew').value });
    $('pwMsg').textContent = r.note;
    setTimeout(() => location.replace('/login'), 1500);
  } catch (err) { $('pwMsg').textContent = err.message; }
};

function showLink(title, r) {
  const life = r.expires_days ? `${r.expires_days} days` : `${r.expires_hours} hours`;
  $('inviteOut').innerHTML = `<p class="lead small">${esc(title)} Send it privately; it works once and expires in ${life}. It is shown only now.</p>
    <div class="linkbox"><code id="linkText">${esc(r.link)}</code><button type="button" id="copyLink">Copy</button></div>`;
  $('copyLink').onclick = () => navigator.clipboard?.writeText(r.link).then(() => { $('copyLink').textContent = 'Copied'; });
}

async function loadTeam() {
  if (state.user?.role !== 'admin') return;
  let p, a;
  try { [p, a] = await Promise.all([api('/api/auth/people'), api('/api/auth/audit')]); }
  catch (err) { $('peopleTable').innerHTML = `<tr><td>${esc(err.message)}</td></tr>`; return; }
  $('peopleTable').innerHTML = `<thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Two-step</th><th>Last sign-in</th><th></th></tr></thead><tbody>${
    p.users.map(u => `<tr><td>${esc(u.name)}</td><td>${esc(u.email)}</td><td>${esc(u.role)}</td><td>${u.two_factor ? 'on' : 'off'}</td>
      <td>${when(u.last_login)}</td><td>${u.id === state.user.id ? '<span class="tag">you</span>' : `
      <button type="button" data-reset="${u.id}">Reset link</button>
      <button type="button" data-dis="${u.id}" data-to="${u.disabled ? 0 : 1}">${u.disabled ? 'Enable' : 'Disable'}</button>`}</td></tr>`).join('')}</tbody>`;
  $('inviteTable').innerHTML = p.invites.length
    ? `<thead><tr><th>Email</th><th>Kind</th><th>Role</th><th>Expires</th><th></th></tr></thead><tbody>${
      p.invites.map(i => `<tr><td>${esc(i.email)}</td><td>${i.purpose === 'reset' ? 'password reset' : 'invite'}</td><td>${esc(i.role)}</td>
        <td>${when(i.expires)}</td><td><button type="button" data-revoke="${esc(i.email)}">Revoke</button></td></tr>`).join('')}</tbody>`
    : '<tr><td class="lead small">No open links.</td></tr>';
  $('auditTable').innerHTML = `<thead><tr><th>When</th><th>Event</th><th>Email</th><th>IP</th><th>Detail</th></tr></thead><tbody>${
    a.events.map(e => `<tr><td>${when(e.at)}</td><td>${esc(e.event.replace(/_/g, ' '))}</td><td>${esc(e.email)}</td><td>${esc(e.ip)}</td><td>${esc(e.detail)}</td></tr>`).join('')}</tbody>`;
}
$('inviteForm').onsubmit = async e => {
  e.preventDefault();
  try {
    const r = await apiPost('/api/auth/invite', { email: $('invEmail').value, role: $('invRole').value });
    showLink(`Invite link for ${$('invEmail').value}.`, r);
    $('invEmail').value = '';
    loadTeam();
  } catch (err) { $('inviteOut').innerHTML = `<p class="err">${esc(err.message)}</p>`; }
};
document.querySelector('main').addEventListener('click', async e => {
  const b = e.target.closest('button[data-reset],button[data-dis],button[data-revoke]');
  if (!b) return;
  try {
    if (b.dataset.reset) {
      showLink('Password reset link.', await apiPost('/api/auth/reset', { user_id: +b.dataset.reset }));
      scrollTo({ top: 0, behavior: 'smooth' });
    } else if (b.dataset.dis) {
      if (b.dataset.to === '1' && !confirm('Disable this account? They are signed out immediately.')) return;
      await apiPost('/api/auth/disable', { user_id: +b.dataset.dis, disabled: b.dataset.to === '1' });
    } else {
      await apiPost('/api/auth/revoke_invite', { email: b.dataset.revoke });
    }
    loadTeam();
  } catch (err) { alert(err.message); }
});

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
    const me = await api('/api/auth/me', 15000);  // a 401 here sends the visitor to /login
    state.user = me.user; state.csrf = me.csrf;
    renderUser();
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
  setInterval(checkServer, 20000);
})();
