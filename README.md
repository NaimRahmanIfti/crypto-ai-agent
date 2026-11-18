# Quant AI Terminal — crypto & stocks

A research-grade market analysis app with a 3D animated web UI. It combines a
**backtested systematic signal** (trend following), a **walk-forward validated
ML model**, and an **analyst desk** that explains *why* — trend, momentum,
volume, relative strength, key levels, chart patterns and news — for the next
session, the next 1–4 weeks and the next 6–12 months.

> Research tool, not financial advice. All performance numbers are
> out-of-sample and net of estimated costs, but past results do not guarantee
> future returns.

## Quick start

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, scikit-learn, lightgbm, requests
python3 server.py                   # http://localhost:8000
```

No API keys needed. Offline UI demo with synthetic prices (clearly labelled
`SYNTHETIC-DEMO`): `QUANT_DEMO=1 python3 server.py`.

## Access: invite only

Nobody can use the terminal, its API or even its page code without an account, and
there is no public sign-up.

1. **First start.** The server prints a one-time link in its terminal
   (`http://localhost:8000/setup#token=…`, valid 1 hour). Open it to create the admin
   account. Only someone who can see the server's terminal can do this; once the admin
   exists the link is dead. If it expires, restart the server for a new one — or skip
   the link entirely and run `python3 scripts/admin.py create` in a second terminal.
   Locked out later? `python3 scripts/admin.py reset you@example.com`.
2. **Invite people.** Admin → **Team** → enter an email → *Create invite link*. Send
   the link privately; it works once, expires in 7 days and can be revoked. The invited
   person picks their own password (12+ characters).
3. **Forgotten password.** Admin → Team → *Reset link* (single use, 24 hours). Using
   it signs that person out everywhere.
4. **Remove someone.** Admin → Team → *Disable*. They are signed out immediately.
5. **Two-step sign-in.** Anyone → **Account** → set up an authenticator app code.

How it is protected (all in `engine/auth.py`, standard library only): scrypt password
hashes; random 256-bit tokens stored only as hashes; HttpOnly + SameSite=Strict session
cookie (12 h idle, 7 days max); a CSRF token on every change; 5 wrong passwords lock the
account for 15 minutes, with a per-IP limit too; the same error for "no such user" and
"wrong password"; Host-header and Origin checks against DNS rebinding and cross-site
posts; strict security headers (CSP, no framing); an audit log of every sign-in,
failure, invite and change (Team → Security log). Data lives in `state/auth.db`
(owner-only permissions, not in git).

**Going beyond your own computer.** The server listens on `127.0.0.1` only. To reach it
from elsewhere, put it behind an HTTPS reverse proxy (Caddy, nginx, Cloudflare Tunnel)
and start it with your domain allowed and cookies marked secure:

```bash
ALLOWED_HOSTS=terminal.example.com SECURE_COOKIES=1 python3 server.py
```

Never expose it over plain HTTP: passwords and session cookies would travel unencrypted.

## What you see

| Panel | What it tells you |
|---|---|
| **Systematic signal · backtested** | Position from the trend-following core: BUY / HOLD LONG, REDUCED LONG or STAY IN CASH, volatility-targeted exposure, chandelier trailing stop, 2R/4R scale-outs, account risk if stopped. |
| **Analyst desk** | Overall verdict + score meter; Next session / Swing / Long-term views with reasons; every bullish, bearish and neutral factor with numbers; key support/resistance; what would invalidate the view; recent headlines with sentiment. |
| **Price & model signal** | Candles, EMA 20/50, and the ML model's out-of-sample P(up) for each bar. |
| **ML accuracy / calibration** | Walk-forward accuracy vs base rate, AUC, p-value, information coefficient, accuracy when confident, reliability plot. |
| **Backtest** | Equity curve and stats for trend core vs ML-only vs blend vs buy & hold. |
| **Universe scan** | Results of `scripts/evaluate.py` across all markets; click a row to open it. |

## Whole-market coverage

- **All crypto:** top 500 coins by market cap (CoinGecko, Binance fallback), live prices, 24h change, volume, market cap.
- **All US stocks:** every listed stock (~7,000) from Nasdaq's screener, filterable by sector.
- **Signals for the whole list:** a background scanner computes quick multi-factor signals for the 500 largest
  stocks and 150 largest coins every 3 hours; any other market gets its full analysis when you open it.

## Every timeframe

Each market gets a 1h / 4h / daily / weekly signal matrix (stocks: 1h / daily / weekly) with a trade plan per
timeframe (entry, 2-ATR stop, 1.5R and 3R targets) and a confluence verdict. Higher timeframes carry more weight:
aligned stacks are the highest-quality setups; conflicts mean trade with the higher timeframe and size down.

## How the bot learns from its own mistakes

1. **Prediction journal.** Every analysis logs what the bot predicted (calibrated ML probability, systematic trend
   position, analyst verdicts, each timeframe's signal) with the price and the time it comes due.
2. **Automatic checking.** A background job resolves due predictions against what the market actually did.
3. **Live track record.** Hit rate and average return of following each kind of call, with the biggest misses and a
   post-mortem of which factors misled it.
4. **Adaptive factor weights.** Each analyst factor's rank correlation with realised returns is measured; factors that
   have been right gain weight (up to 1.8x), factors that keep misleading lose weight (down to 0.3x). Neutral until 40
   checked calls, shrunk towards neutral while the sample is small.
5. **Probability calibration.** The ML probability is Platt-calibrated on each market's own out-of-sample history, so
   "60%" means what it says.

This is measured, out-of-sample learning, not a guarantee of accuracy. Markets are close to efficient; the journal
exists so you can see exactly how good the signals are, live.

## News

Breaking-news feed (geopolitics, economy, companies and markets, crypto) from Google News, BBC, CNBC, MarketWatch,
CoinDesk and Cointelegraph: last 48 hours, newest first, syndicated duplicates removed, refreshed every 3 minutes.
Per-market news adds Yahoo Finance (stocks) and coin-filtered crypto feeds.

## Research desk (20 agents)

Open the **Agents** view and press *Run the desk*. Each agent has one job and one skill:

| Stage | Agents (skill) |
|---|---|
| Route | Navigator (trading-skills-navigator) |
| Discover | Scanner (trader-scanner), Signal (trading-signals), Breakout planner (breakout-trade-planner), Pairs (pair-trade-screener), Earnings reaction (earnings-trade-analyzer) |
| Context | News (transformers), Fundamentals (trader-analyst), Congress (trader-pelosi) |
| Risk | Replay (trader-timemachine), Sizing (trader-budget), Audit (backward-traceability), **Discipline gate** (pre-trade-discipline-gate) |
| Act | Broker (trader-broker, tickets only), Alerts (trader-alerts) |
| Learn | Memory (trader-memory-core), Coach (trade-performance-coach), Research (trade-hypothesis-ideator), Pivot (strategy-pivot-designer) |
| Present | Dashboard (trader-dashboard) |

- **Every signal goes through the discipline gate** before it is shown: GO, review first, blocked, or watchlist,
  with the reasons. The gate journals each decision to `state/journal/pre-trade-discipline/`.
- **Nothing places an order.** The Broker agent writes tickets you place yourself in your broker's app.
- **Risk rules** live in `config/risk.json` (created on first run): $10,000 budget, $100 max loss per trade,
  20% position cap, 10% cash buffer, 4% total open risk, 24h revenge window, daily/weekly loss halts.
- **Optional upgrades:** `SEC_USER_AGENT="Your Name you@example.com"` (filed numbers from SEC EDGAR),
  `X70_API_KEY` (trading-signals second opinion), `pip install transformers torch` (FinBERT sentiment),
  `pdftotext` (tickers from Congress filings). Everything works without them.
- Record fills and exits in *Your theses*; the Coach reviews closed trades and the gate's revenge window and
  circuit breaker use them.

## Demo trading: does the signal actually work?

The **Demo trades** tab answers that with trades, not opinions. While the server runs,
the desk runs by itself every hour (crypto always, stocks while the US market is open) and
every idea the discipline gate rules on becomes a paper trade with real prices:

- **Demo account:** GO ideas only, in a $10,000 paper account under the same risk rules as
  real money ($100 max loss per trade, 4% total open risk, position cap, cash buffer).
- **Shadow book:** ideas the gate sent to review or blocked are traded too, with no money,
  so you can see whether blocking them actually avoided losers ("Is the gate right?").
- **Honest fills:** market entries at the live quote (stocks only while the market is
  open); breakouts wait for their pivot; exits from hourly highs/lows after the fill; if a
  bar touches both stop and target, the stop counts; gaps through the stop exit at the worse
  price; 0.15% (crypto) / 0.05% (stocks) fees and slippage per side; time exit at the
  plan's horizon.
- **Promised vs delivered:** the replay's predicted target-hit rate and average R against
  what the demo trades really did.
- **It learns:** after 20 closed demo trades, a setup that is losing (average below 0R) is
  sent to *review first*; after 40 trades with a clear loss (≤ −0.2R, t ≤ −2) the gate
  **blocks** it. Its shadow trades keep running, so a setup that recovers is let back in.
  Winning setups show as "real edge". Thresholds: `paper_min_trades`,
  `paper_block_trades`, `paper_every_minutes` in `config/risk.json`.

**Every style, both directions.** The desk now takes:

| Style | Timeframes | Held for | Long | Short |
|---|---|---|---|---|
| Scalp | 1m, 3m, 5m | 30 min to 2 h | pullback, breakout, RSI fade | pullback, breakdown, RSI fade |
| Day trade | 15m (30m/1h reversals) | 4 to 6 h | same | same |
| Short swing | 4h | ~2 days | same | same |
| Swing | daily | 1 to 2 weeks | breakout, bull flag, reclaimed MA20, oversold | breakdown, bear flag, lost MA20, overbought |

The scalp lane rescans 1m–15m charts every 5 minutes; the full desk runs hourly. Stock day
trades are closed by 15:55 New York time. Where the bot's own research found **no edge**
(1m/3m/15m/4h, and daily shorts, which turned the backtest from profit to loss), the gate
sends the idea to *review first*, so it is demo-traded in the **shadow book**. Once its own
record shows an edge (20+ closed trades, t ≥ 2), it **graduates** to GO and the demo account.
The gate also flags any trade whose stop is so tight that fees and slippage would cost more
than 0.5R, which is most 1-minute scalps.

History lives in `state/paper.db`. Turn the autopilot off with the switch on the tab.
Small samples are mostly luck: judge a setup after 30+ closed trades, not 5.

## How it works

```
engine/data.py      market data: Binance → Coinbase → Kraken (crypto); Nasdaq → Yahoo (stocks; Yahoo also serves 1h); disk cache
engine/features.py  ~50 stationary features (vol-normalised returns, trend, oscillators, candles, volume flow, vs-benchmark)
engine/model.py     logistic regression + regularised LightGBM ensemble; purged walk-forward validation
engine/backtest.py  next-bar execution, costs, volatility targeting, Sharpe/Sortino/drawdown/probabilistic Sharpe
engine/signals.py   systematic trade plan (trend core + ML tilt)
engine/analyst.py   multi-horizon reasoning: factors, levels, patterns, session clock, invalidation
engine/news.py      Google News RSS headlines + finance lexicon sentiment
server.py           zero-dependency HTTP server (JSON API + static UI)
web/                Three.js 3D scene, canvas charts, glass UI
```

**Honesty rules built into the engine** (and enforced by `tests/`):

- Features at bar *t* only use data up to *t*; the forming candle is dropped.
- Validation is walk-forward with a purge gap — never shuffled cross-validation.
- The model must show ~50% accuracy on pure noise and must find a planted signal.
- Backtests charge costs (crypto 10 bps, stocks 3 bps per unit turnover).

## What the research found (real Binance daily data, 2017–2026, 10 coins)

| Strategy (mean of 10 markets, OOS, net) | Sharpe | CAGR | Max DD | Beats B&H |
|---|---|---|---|---|
| ML model long/short | 0.07 | −2.4% | −55% | 3/10 |
| ML model long/flat | 0.46 | 8.2% | −36% | 5/10 |
| **Trend following (TSMOM 20/60/120, long/flat)** | **0.83** | **19.3%** | **−32%** | **7/10** |
| 50/50 trend + ML gate | 0.74 | 16.9% | −34% | 8/10 |
| Buy & hold | 0.60 | 17.6% | −84% | — |

The ML model's next-day edge is real but small (51.6% accuracy vs 51.2% base
rate, AUC 0.519, significant on 5/10 coins). That is normal for daily markets
and too small to trade alone, so the **trend core drives positions** and the ML
model is used as a short-term timing tilt. Six variants were compared, so expect
live results somewhat below the backtest.

## Scripts

```bash
python3 scripts/evaluate.py --intervals 1d --horizons 1      # full universe → reports/evaluation.{md,json}
python3 scripts/research.py crypto                           # strategy variant comparison on cached data
python3 -m unittest discover tests                           # correctness tests
```

## API

```
GET /api/universe
GET /api/analyze?symbol=BTC&asset=crypto&interval=1d&horizon=1   (interval: crypto 1h|4h|1d, stock 1h|1d)
GET /api/quote?symbol=AAPL&asset=stock
GET /api/report
```

Every endpoint needs a signed-in session (cookie); anything that changes data is a POST
carrying the `X-CSRF-Token` header from `GET /api/auth/me`. Without a session the API
answers `401 {"signin": true}`.

## Limits

- Stock fundamentals (earnings, valuation, guidance) are not modelled.
- News sentiment is a headline word list — transparent but crude.
- The analyst desk's rule factors are standard practice but not individually
  backtested; only the trend core and the ML model are.
- Free data sources can rate-limit (Yahoo returns HTTP 429 under load).
