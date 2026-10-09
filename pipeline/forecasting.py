"""Prefix-only pooled forecasting and auditable publication-time news features."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ranked_ids(panel, seed, excluded=()):
    coverage = panel.iloc[:, :12].notna().sum(axis=1).ge(9)
    groups = coverage.groupby(level=0).agg(["all", "count"])
    ids = [int(x) for x in groups.index[groups["all"] & groups["count"].eq(6)]
           if int(x) not in excluded]
    return sorted(ids, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())


def fill_history(history):
    """No interpolation across the forecast boundary; leading gaps use prefix mean."""
    a = np.asarray(history, dtype=float)
    if a.ndim != 2 or a.shape[1] < 1 or not np.isfinite(a).any(axis=1).all():
        raise ValueError("Every series needs observed history")
    out = pd.DataFrame(a).ffill(axis=1).to_numpy()
    means = np.nanmean(a, axis=1)
    return np.where(np.isfinite(out), out, means[:, None])


def news_features(events, as_of):
    """As-of join on availability, never on event or target date."""
    as_of = pd.Timestamp(as_of)
    known = events[pd.to_datetime(events.available_at) < as_of].sort_values("available_at")
    if known.empty:
        return np.zeros(4)
    recent = known[pd.to_datetime(known.available_at) >= as_of-pd.Timedelta(days=90)]
    changed = known[known.delta_pp != 0]
    age = min((as_of-pd.Timestamp(changed.iloc[-1].available_at)).days, 365) if len(changed) else 365
    return np.array([known.iloc[-1].rate/25, recent.delta_pp.sum()/10,
                     (recent.delta_pp != 0).sum()/5, age/365], dtype=float)


def features(history, category_codes, category_count, start_month, horizon, events=None, as_of=None):
    a = fill_history(history)
    n, c = a.shape
    y = np.log1p(np.maximum(a, 0))
    last = y[:, -1]
    target = c + horizon - 1
    seasonal_idx = target-12
    seasonal = y[:, seasonal_idx]-last if 0 <= seasonal_idx < c else np.zeros(n)
    yoy = np.median(y[:, 12:]-y[:, :-12], axis=1) if c>12 else np.zeros(n)
    month = (start_month-1+target) % 12
    recent = y[:, -min(c, 6):]
    x = np.arange(recent.shape[1], dtype=float)
    xc = x-x.mean()
    slope = recent @ xc / max(float(xc@xc), 1)
    cat = np.eye(category_count)[category_codes]
    harmonic = np.array([np.sin(2*np.pi*k*month/12) for k in (1,2)] +
                        [np.cos(2*np.pi*k*month/12) for k in (1,2)])
    cols = [last, y[:, -min(c,2)]-last, y[:, -min(c,3)]-last,
            np.mean(y[:, -3:], axis=1)-last, np.mean(recent,axis=1)-last,
            slope, slope*min(horizon,6), seasonal, yoy,
            np.full(n, horizon/12), np.full(n, np.log1p(horizon)),
            np.full(n, float(0 <= seasonal_idx < c)), np.full(n, c/24)]
    parts = [np.column_stack(cols), cat, np.tile(harmonic,(n,1)),
             (cat[:,:,None]*harmonic[None,None,:]).reshape(n,-1)]
    if events is not None:
        v = news_features(events, as_of)
        parts += [np.tile(v,(n,1)), (cat[:,:,None]*v[None,None,:]).reshape(n,-1)]
    return np.concatenate(parts,axis=1), a[:, -1]


class PooledRidge:
    def __init__(self, config, categories, events=None):
        self.config, self.categories, self.events = config, list(categories), events
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=config["ridge_alpha"]))

    def _x(self, history, codes, dates, horizon):
        as_of = dates[0] + pd.DateOffset(months=history.shape[1])
        return features(history, codes, len(self.categories), dates[0].month,
                        horizon, self.events, as_of)

    def fit(self, history, codes, dates):
        xs, ys, weights = [], [], []
        a = np.asarray(history)
        for c in range(6, a.shape[1]):
            for h in self.config["horizons"]:
                t = c+h-1
                if t >= a.shape[1]:
                    continue
                valid = np.isfinite(a[:, :c]).any(axis=1) & np.isfinite(a[:,t])
                if not valid.any():
                    continue
                X, last = self._x(a[valid,:c],codes[valid],dates,h)
                xs.append(X)
                ys.append(np.log1p(a[valid,t])-np.log1p(last))
                weights.append(np.sqrt(np.maximum(last,1)))
        if not xs:
            raise ValueError("Insufficient pooled training history")
        w = np.concatenate(weights)
        self.model.fit(np.concatenate(xs),np.concatenate(ys),ridge__sample_weight=w/w.mean())
        self.training_rows = len(w)
        self.training_months = a.shape[1]
        return self

    def predict(self, history, codes, dates, horizon):
        X, last = self._x(np.asarray(history),codes,dates,horizon)
        lo, hi = self.config["prediction_ratio_clip"]
        change = np.clip(self.model.predict(X), np.log(lo), np.log(hi))
        return np.maximum(np.expm1(np.log1p(last)+change),0)
