"""Prefix-only panel forecasting models."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def fill(a):
    a = np.asarray(a, float).copy()
    if a.ndim != 2 or a.shape[1] < 1 or not np.isfinite(a).any(axis=1).all():
        raise ValueError('Each row requires observed prefix history')
    means = np.nanmean(a, axis=1)
    for t in range(a.shape[1]):
        missing = ~np.isfinite(a[:, t])
        a[missing, t] = means[missing] if t == 0 else a[missing, t-1]
    return a


def factors(history, codes):
    z = np.log1p(np.maximum(fill(history), 0))
    f = np.empty_like(z)
    for cat in np.unique(codes):
        m = codes == cat
        f[m] = np.median(z[m], axis=0)
    return z, f, z-f


def early_growth(f):
    """A regularized trend, not an externally measured inflation/growth prior."""
    t = np.arange(len(f), dtype=float)
    X = np.column_stack([np.ones(len(t)), (t-t.mean())/12,
        np.sin(2*np.pi*t/12), np.cos(2*np.pi*t/12),
        np.sin(4*np.pi*t/12), np.cos(4*np.pi*t/12)])
    penalty = np.diag([0, .2, .05, .05, .05, .05])
    beta = np.linalg.solve(X.T@X+penalty, X.T@f)
    return float(np.clip(beta[1], -.2, .7))


def features(d, codes, h):
    c = d.shape[1]
    last = d[:, -1]
    recent = d[:, -min(6,c):]
    t = np.arange(recent.shape[1]); xc = t-t.mean()
    slope = recent@xc/max(xc@xc, 1)
    month = (c+h-1)%12
    return np.column_stack([last, d[:, -min(2,c)]-last, d[:, -min(3,c)]-last,
        recent.mean(axis=1)-last, slope, slope*min(h,6),
        np.full(len(d), h), np.full(len(d), np.sin(2*np.pi*month/12)),
        np.full(len(d), np.cos(2*np.pi*month/12)), np.eye(6)[codes]])


class PanelModel:
    def __init__(self, history, codes, boost=True):
        self.codes = np.asarray(codes)
        self.z, self.f, self.d = factors(history, self.codes)
        self.booster = None
        if boost:
            # Every label and every factor uses only this origin's history.
            xs, ys, ws = [], [], []
            take = np.arange(0, len(history), 3)
            for c in range(6, self.z.shape[1]):
                for h in (1,3,6,12):
                    target = c+h-1
                    if target >= self.z.shape[1]:
                        continue
                    xs.append(features(self.d[take,:c], self.codes[take], h))
                    ys.append(self.d[take,target]-self.d[take,c-1])
                    ws.append(np.sqrt(np.expm1(self.z[take,c-1]).clip(1)))
            if xs:
                w = np.concatenate(ws)
                self.booster = HistGradientBoostingRegressor(loss='absolute_error',
                    max_iter=70, max_leaf_nodes=15, learning_rate=.06,
                    l2_regularization=10, min_samples_leaf=40, early_stopping=False,
                    random_state=20261009)
                self.booster.fit(np.concatenate(xs), np.concatenate(ys),
                                 sample_weight=w/w.mean())

    def predict(self, h):
        z, f, d = self.z, self.f, self.d
        c = z.shape[1]; target = c+h-1-12
        if not 0 <= target < c:
            raise ValueError('Need an observed seasonal anchor')
        panel_growth = np.empty(len(z))
        local_growth = np.empty(len(z))
        for cat in np.unique(self.codes):
            m = self.codes == cat; cf = f[m][0]
            pg = np.median((cf[12:]-cf[:-12])[-3:]) if c>12 else early_growth(cf)
            panel_growth[m] = np.clip(pg, -.2, .7)
            local_growth[m] = (np.median((z[m,12:]-z[m,:-12])[:,-3:],axis=1)
                               if c>12 else pg)
        national = f[:,target]+panel_growth
        state = .7*d[:,-1]+.3*np.median(d[:,-3:],axis=1)
        damped_state = (.8**h)*state+(1-.8**h)*d[:,target]
        out = {
            'seasonal': z[:,target],
            'pooled_growth': z[:,target]+panel_growth,
            'shrink_growth': z[:,target]+.75*panel_growth+.25*local_growth,
            'factor_state': national+state,
            'factor_damped': national+damped_state,
        }
        delta = np.zeros(len(z)) if self.booster is None else self.booster.predict(features(d,self.codes,h))
        out['factor_boost'] = national+d[:,-1]+np.clip(delta,-.2,.2)
        return {name:np.maximum(np.expm1(v),0) for name,v in out.items()}


MEMBERS = ['pooled_growth','shrink_growth','factor_state','factor_damped','factor_boost']


def mixture_weights(history, cat, h, origin):
    # history holds only aggregate errors from earlier issued forecasts.
    mature = [r for r in history if r['category']==cat and r['horizon']==h
              and r['target_index']<origin]
    if len(mature)<3:
        return np.ones(len(MEMBERS))/len(MEMBERS)
    loss = np.array([[r[n] for n in MEMBERS] for r in mature[-6:]]).mean(axis=0)
    weights = 1/np.maximum(loss,1)**2
    return weights/weights.sum()
