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

## Limits

- Stock fundamentals (earnings, valuation, guidance) are not modelled.
- News sentiment is a headline word list — transparent but crude.
- The analyst desk's rule factors are standard practice but not individually
  backtested; only the trend core and the ML model are.
- Free data sources can rate-limit (Yahoo returns HTTP 429 under load).
