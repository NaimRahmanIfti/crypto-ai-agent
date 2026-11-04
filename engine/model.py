"""Direction model + purged walk-forward validation.

The model predicts P(close[t+h] > close[t]) from features at t.
Ensemble = L2 logistic regression (stable, linear) + shallow, heavily
regularised LightGBM (non-linear interactions). Averaging the two reduces
variance, which matters far more than raw capacity on noisy market data.

Validation is walk-forward only: train on [0, k), purge `horizon` bars, test on
[k+horizon, k+horizon+step), roll forward. No shuffled CV - it leaks the future
through overlapping labels and autocorrelated features.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional
    lgb = None

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelConfig:
    horizon: int = 1
    min_train: int = 750       # bars before the first out-of-sample prediction
    step: int = 60             # retrain cadence (bars)
    max_train: int | None = 4000  # rolling cap so stale regimes fade out
    use_gbm: bool = True
    seed: int = 7


class DirectionModel:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.lin = None
        self.gbm = None
        self.cols: list[str] = []
        self.med: pd.Series | None = None

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.cols].fillna(self.med)
        # Winsorise to training quantiles - fat tails otherwise dominate the linear model.
        return np.clip(X.values, self.lo, self.hi)

    def fit(self, X: pd.DataFrame, fwd: pd.Series):
        self.cols = list(X.columns)
        self.med = X.median()
        self.lo, self.hi = X.quantile(0.005).values, X.quantile(0.995).values
        Xp = self._prep(X)
        y = (fwd.values > 0).astype(int)
        # Weight by move size (in vol units): big moves are what P&L is made of,
        # tiny moves are mostly noise around zero.
        scale = np.nanmedian(np.abs(fwd.values)) or 1.0
        w = np.clip(np.abs(fwd.values) / scale, 0.2, 4.0)

        self.lin = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        self.lin.fit(Xp, y, logisticregression__sample_weight=w)
        if self.cfg.use_gbm and lgb is not None:
            self.gbm = lgb.LGBMClassifier(
                n_estimators=250, learning_rate=0.03, num_leaves=15, max_depth=4,
                min_child_samples=max(50, len(y) // 40), subsample=0.7, subsample_freq=1,
                colsample_bytree=0.6, reg_lambda=10.0, reg_alpha=1.0,
                random_state=self.cfg.seed, verbose=-1, n_jobs=1)
            self.gbm.fit(Xp, y, sample_weight=w)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xp = self._prep(X)
        p = self.lin.predict_proba(Xp)[:, 1]
        if self.gbm is not None:
            p = 0.5 * p + 0.5 * self.gbm.predict_proba(Xp)[:, 1]
        return p

    def feature_importance(self, top: int = 12) -> list[dict]:
        coef = np.abs(self.lin[-1].coef_[0])
        imp = coef / coef.sum()
        if self.gbm is not None:
            g = self.gbm.booster_.feature_importance("gain").astype(float)
            imp = 0.5 * imp + 0.5 * g / max(g.sum(), 1e-12)
        order = np.argsort(-imp)[:top]
        return [{"feature": self.cols[i], "importance": float(imp[i])} for i in order]


@dataclass
class WalkForwardResult:
    proba: pd.Series            # out-of-sample P(up), indexed by bar
    fwd: pd.Series              # realised forward return for those bars
    metrics: dict = field(default_factory=dict)


def walk_forward(X: pd.DataFrame, fwd: pd.Series, cfg: ModelConfig) -> WalkForwardResult:
    data = X.join(fwd.rename("__fwd")).dropna(subset=["__fwd"])
    data = data[data.drop(columns="__fwd").notna().mean(axis=1) > 0.8]
    Xa, ya = data.drop(columns="__fwd"), data["__fwd"]
    n, h = len(data), cfg.horizon
    preds = pd.Series(np.nan, index=data.index)
    folds, k = [], cfg.min_train
    while k + h < n:
        tr_end = k                     # train rows [tr_start, tr_end)
        tr_start = 0 if cfg.max_train is None else max(0, tr_end - cfg.max_train)
        folds.append((tr_start, tr_end, k + h, min(k + h + cfg.step, n)))
        k += cfg.step

    def run(f):
        tr_start, tr_end, te_start, te_end = f
        # Purge: the last h training labels overlap the test window, drop them.
        m = DirectionModel(cfg).fit(Xa.iloc[tr_start:tr_end - h], ya.iloc[tr_start:tr_end - h])
        return te_start, te_end, m.predict_proba(Xa.iloc[te_start:te_end])

    # Folds are independent; LightGBM and BLAS release the GIL, so threads parallelise well.
    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        for te_start, te_end, p in ex.map(run, folds):
            preds.iloc[te_start:te_end] = p
    ok = preds.notna()
    res = WalkForwardResult(preds[ok], ya[ok])
    res.metrics = classification_metrics(res.proba, res.fwd, h)
    return res


def classification_metrics(p: pd.Series, fwd: pd.Series, horizon: int = 1) -> dict:
    y = (fwd > 0).astype(int).values
    pv = p.values
    pred = (pv > 0.5).astype(int)
    n = len(y)
    acc = float((pred == y).mean())
    base = float(max(y.mean(), 1 - y.mean()))  # best constant guess
    # Overlapping labels (h>1) are not independent: use effective sample size n/h.
    n_eff = max(1, n // max(1, horizon))
    p_val = float(stats.binomtest(int(round(acc * n_eff)), n_eff, 0.5, alternative="greater").pvalue)
    p_val_base = float(stats.binomtest(int(round(acc * n_eff)), n_eff, base, alternative="greater").pvalue)
    out = {
        "n": n, "accuracy": acc, "baseline_accuracy": base, "up_rate": float(y.mean()),
        "edge_vs_baseline": acc - base,
        "auc": float(roc_auc_score(y, pv)) if 0 < y.sum() < n else 0.5,
        "brier": float(brier_score_loss(y, pv)),
        "brier_baseline": float(brier_score_loss(y, np.full(n, y.mean()))),
        "log_loss": float(log_loss(y, np.clip(pv, 1e-6, 1 - 1e-6))),
        "ic": float(stats.spearmanr(pv, fwd.values).statistic),
        "p_value_vs_coinflip": p_val, "p_value_vs_baseline": p_val_base,
        "confident": [],
        "calibration": [],
    }
    # Accuracy when the model is confident: what a selective trader actually experiences.
    for t in (0.02, 0.05, 0.08, 0.12):
        m = np.abs(pv - 0.5) >= t
        if m.sum() >= 30:
            out["confident"].append({"threshold": t, "coverage": float(m.mean()),
                                     "accuracy": float((pred[m] == y[m]).mean()), "n": int(m.sum())})
    bins = np.clip(((pv - 0.3) / 0.4 * 8).astype(int), 0, 7)
    for b in range(8):
        m = bins == b
        if m.sum() >= 20:
            out["calibration"].append({"p_mean": float(pv[m].mean()), "up_rate": float(y[m].mean()), "n": int(m.sum())})
    return out


def momentum_baseline(fwd: pd.Series, past_ret: pd.Series) -> float:
    """Accuracy of 'next move = sign of last move' - the naive benchmark to beat."""
    idx = fwd.index.intersection(past_ret.dropna().index)
    return float(((past_ret[idx] > 0) == (fwd[idx] > 0)).mean())
