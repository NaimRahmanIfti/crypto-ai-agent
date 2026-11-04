// Small dependency-free canvas charts.
const C = { up: '#22e3a0', down: '#ff4d6d', accent: '#6c8cff', amber: '#ffb547', grid: 'rgba(140,155,255,0.08)', text: '#8b95c9' };

function setup(canvas) {
  const r = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2);
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
  return { ctx, w: r.width, h: r.height };
}
const fmt = v => Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(4);
const ema = (xs, n) => { const k = 2 / (n + 1); let e = xs[0]; return xs.map(x => (e = x * k + e * (1 - k))); };

// Candles + EMA20/50 + model P(up) strip, with crosshair.
export function candleChart(canvas, bars, proba, onHover) {
  const draw = (hx = null) => {
    const { ctx, w, h } = setup(canvas);
    const data = bars.slice(-160); if (!data.length) return;
    const padR = 64, mainH = h * 0.76, stripY = mainH + 14, stripH = h - stripY - 16;
    const lo = Math.min(...data.map(b => b.l)), hi = Math.max(...data.map(b => b.h));
    const y = v => 8 + (1 - (v - lo) / (hi - lo || 1)) * (mainH - 16);
    const bw = (w - padR) / data.length, x = i => i * bw + bw / 2;
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.text; ctx.lineWidth = 1;
    for (let k = 0; k <= 5; k++) {
      const v = lo + (hi - lo) * k / 5, yy = y(v);
      ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
      ctx.fillText(fmt(v), w - padR + 6, yy + 4);
    }
    data.forEach((b, i) => {
      const col = b.c >= b.o ? C.up : C.down;
      ctx.strokeStyle = col; ctx.fillStyle = col;
      ctx.beginPath(); ctx.moveTo(x(i), y(b.h)); ctx.lineTo(x(i), y(b.l)); ctx.stroke();
      const top = y(Math.max(b.o, b.c)), bh = Math.max(1, Math.abs(y(b.o) - y(b.c)));
      ctx.globalAlpha = 0.9; ctx.fillRect(x(i) - bw * 0.35, top, bw * 0.7, bh); ctx.globalAlpha = 1;
    });
    const closes = bars.map(b => b.c), off = bars.length - data.length;
    [[20, C.accent], [50, C.amber]].forEach(([n, col]) => {
      const e = ema(closes, n).slice(off);
      ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.85; ctx.beginPath();
      e.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))); ctx.stroke(); ctx.globalAlpha = 1;
    });
    // last price tag
    const last = data[data.length - 1].c;
    ctx.fillStyle = last >= data[data.length - 1].o ? C.up : C.down;
    ctx.fillRect(w - padR + 2, y(last) - 9, padR - 4, 18);
    ctx.fillStyle = '#05060f'; ctx.fillText(fmt(last), w - padR + 6, y(last) + 4);

    // probability strip
    const pmap = new Map(proba.map(p => [p.t, p.p]));
    ctx.strokeStyle = C.grid; ctx.strokeRect(0, stripY, w - padR, stripH);
    const mid = stripY + stripH / 2;
    ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w - padR, mid); ctx.stroke(); ctx.setLineDash([]);
    data.forEach((b, i) => {
      const p = pmap.get(b.t); if (p === undefined) return;
      const hgt = (p - 0.5) / 0.25 * (stripH / 2);
      ctx.fillStyle = p >= 0.5 ? C.up : C.down; ctx.globalAlpha = 0.25 + Math.min(0.75, Math.abs(p - 0.5) * 6);
      ctx.fillRect(x(i) - bw * 0.4, mid - Math.max(-stripH / 2, Math.min(stripH / 2, hgt)), bw * 0.8, Math.max(-stripH / 2, Math.min(stripH / 2, hgt)));
      ctx.globalAlpha = 1;
    });
    ctx.fillStyle = C.text; ctx.fillText('model P(up), out-of-sample', 6, stripY + 12);

    if (hx !== null) {
      const i = Math.max(0, Math.min(data.length - 1, Math.floor(hx / bw)));
      ctx.strokeStyle = 'rgba(200,210,255,0.35)'; ctx.beginPath(); ctx.moveTo(x(i), 0); ctx.lineTo(x(i), h); ctx.stroke();
      onHover && onHover(data[i], pmap.get(data[i].t));
    }
  };
  draw();
  canvas.onmousemove = e => draw(e.offsetX);
  canvas.onmouseleave = () => { draw(); onHover && onHover(null); };
  return draw;
}

export function equityChart(canvas, eq) {
  const { ctx, w, h } = setup(canvas);
  if (!eq.length) return;
  const vals = eq.flatMap(e => [e.s, e.b]).map(Math.log);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const y = v => 10 + (1 - (Math.log(v) - lo) / (hi - lo || 1)) * (h - 30), x = i => i / (eq.length - 1) * (w - 50);
  ctx.strokeStyle = C.grid; ctx.fillStyle = C.text;
  [0.5, 1, 2, 4, 8, 16, 32].forEach(v => {
    if (Math.log(v) < lo || Math.log(v) > hi) return;
    ctx.beginPath(); ctx.moveTo(0, y(v)); ctx.lineTo(w - 50, y(v)); ctx.stroke(); ctx.fillText(`${v}x`, w - 44, y(v) + 4);
  });
  const line = (key, col, fill) => {
    ctx.beginPath(); eq.forEach((e, i) => i ? ctx.lineTo(x(i), y(e[key])) : ctx.moveTo(x(i), y(e[key])));
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
    if (fill) {
      ctx.lineTo(x(eq.length - 1), h - 20); ctx.lineTo(0, h - 20); ctx.closePath();
      const g = ctx.createLinearGradient(0, 0, 0, h); g.addColorStop(0, 'rgba(108,140,255,0.35)'); g.addColorStop(1, 'rgba(108,140,255,0)');
      ctx.fillStyle = g; ctx.fill();
    }
  };
  line('b', 'rgba(139,149,201,0.7)'); line('s', C.accent, true);
  const yr = t => new Date(t * 1000).getFullYear();
  ctx.fillStyle = C.text; ctx.fillText(yr(eq[0].t), 0, h - 4); ctx.fillText(yr(eq[eq.length - 1].t), w - 90, h - 4);
}

export function calibrationChart(canvas, cal) {
  const { ctx, w, h } = setup(canvas);
  const p = 24, x = v => p + (v - 0.3) / 0.4 * (w - 2 * p), y = v => h - p - (v - 0.3) / 0.4 * (h - 2 * p);
  ctx.strokeStyle = C.grid; ctx.strokeRect(p, p, w - 2 * p, h - 2 * p);
  ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(139,149,201,0.5)';
  ctx.beginPath(); ctx.moveTo(x(0.3), y(0.3)); ctx.lineTo(x(0.7), y(0.7)); ctx.stroke(); ctx.setLineDash([]);
  const maxN = Math.max(...cal.map(c => c.n), 1);
  cal.forEach(c => {
    ctx.fillStyle = C.accent; ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.arc(x(c.p_mean), y(Math.max(0.3, Math.min(0.7, c.up_rate))), 3 + 7 * Math.sqrt(c.n / maxN), 0, 7); ctx.fill();
  });
  ctx.globalAlpha = 1; ctx.fillStyle = C.text;
  ctx.fillText('predicted →', w - p - 70, h - 6); ctx.save(); ctx.translate(10, p + 60); ctx.rotate(-Math.PI / 2); ctx.fillText('realised ↑', 0, 0); ctx.restore();
}
