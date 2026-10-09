"""V5 statistical correction of a frozen V4 forecast; no fitting on future facts."""
import numpy as np
import pandas as pd
import json
from pathlib import Path

def load_config(path=None):
    config=json.loads(Path(path or Path(__file__).with_name('config.json')).read_text())
    if set(config)!={'growth_window','level_window','growth_clip','local_growth_clip','pooled_growth_weight'}:
        raise ValueError('Unknown or missing V5 configuration keys')
    for key in ['growth_window','level_window']:
        if type(config[key]) is not int or not 1<=config[key]<=12:
            raise ValueError('Windows must be integers from 1 to 12')
    for key in ['growth_clip','local_growth_clip']:
        pair=config[key]
        if not isinstance(pair,list) or len(pair)!=2 or not all(type(v) in (int,float) and np.isfinite(v) for v in pair) or pair[0]>=pair[1]:
            raise ValueError('Clipping bounds must be finite and ordered')
    weight=config['pooled_growth_weight']
    if type(weight) not in (int,float) or not np.isfinite(weight) or not 0<=weight<=1:
        raise ValueError('Weight must be between zero and one')
    return config

def fill_prefix(history):
    x=np.asarray(history,dtype=float)
    if x.ndim!=2 or x.shape[1]<12 or not np.isfinite(x).any(axis=1).all():
        raise ValueError('Each series needs observed history and a 12-month calendar')
    return np.maximum(pd.DataFrame(x.T).ffill().bfill().to_numpy().T,1.0)

def components(history,categories,national,origin,horizon,lag=1,config=None):
    config=load_config() if config is None else config
    origin=pd.Period(origin,freq='M')
    if lag<1 or horizon not in [1,3,6,12]:raise ValueError('Invalid lag/horizon')
    y=fill_prefix(history);z=np.log(y);c=y.shape[1];target=origin+horizon
    season_index=c+horizon-13
    if not 0<=season_index<c:raise ValueError('Seasonal anchor outside history')
    n=national.loc[:origin-lag].copy()
    out={k:np.zeros(len(y)) for k in ['national_level3','local_blend','panel_recent']}
    categories=np.asarray(categories)
    for cat in sorted(set(categories)):
        mask=categories==cat;yy=y[mask];zz=z[mask]
        field={'Продовольствие':'food','Общественное питание':'catering'}.get(cat,'total')
        ns=n[field]
        growth=float(np.clip(np.median(np.log(ns/ns.shift(12)).dropna().iloc[-config['growth_window']:]),*config['growth_clip']))
        if not np.isfinite(growth):raise ValueError('National history too short')
        anchor=target-12
        while anchor not in ns.index:anchor-=12
        nt=float(ns.loc[anchor]*np.exp(growth*(target.ordinal-anchor.ordinal)/12))
        months=pd.period_range(end=origin,periods=c,freq='M')
        nh=np.array([ns.loc[m] if m in ns.index else ns.loc[m-12]*np.exp(growth) for m in months])
        ratios=yy/nh
        level1=ratios[:,-1]*nt;level3=np.median(ratios[:,-config['level_window']:],axis=1)*nt
        blend=(level1+level3)/2
        if c>12:
            local_growth=np.median(zz[:,12:]-zz[:,:-12],axis=1)
            cg=np.median(local_growth)
            # Same latest-three growth policy as the first experiment.
            lg3=np.median((zz[:,12:]-zz[:,:-12])[:,-config['growth_window']:],axis=1)
            weight=config['pooled_growth_weight']
            seasonal=yy[:,season_index]*np.exp(np.clip(weight*np.median(lg3)+(1-weight)*lg3,*config['local_growth_clip']))
            f=np.median(zz,axis=0)
            panel=np.exp(f[season_index]+(f[-1]-f[-13])+zz[:,-1]-f[-1])
        else:
            seasonal=yy[:,season_index]*np.exp(growth)
            panel=(seasonal+blend)/2
        out['national_level3'][mask]=level3
        out['local_blend'][mask]=(seasonal+blend)/2
        out['panel_recent'][mask]=panel
    return out

def forecast(history,categories,national,origin,horizon,baseline,lag=1,config=None):
    baseline=np.asarray(baseline,float)
    if not np.isfinite(baseline).all():raise ValueError('Non-finite V4 baseline')
    if horizon in [3,6]:return baseline.copy()
    p=components(history,categories,national,origin,horizon,lag,config)
    if horizon==12:return p['national_level3']
    return (baseline+p['panel_recent']+p['local_blend'])/3
