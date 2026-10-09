"""Panel-factor forecasting experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import baseline_forecasts
from forecasting import PooledRidge, fill_history
from experiment import ROOT, prepare, foundation_pipeline, metric_table, write_json
from upgrade_v2 import adapted_foundation, candidates, apply_policy, quantile, robust_chart
from detection import innovations

OUT = ROOT / 'v3_results'
KEY = ['territory_id', 'category', 'origin', 'horizon']

def panel_candidates(history, codes, h):
    """All inputs are observed prefixes. Cross-sectional factor is not national spending."""
    valid = np.isfinite(history).any(axis=1)
    if not valid.all():
        partial = panel_candidates(history[valid], codes[valid], h)
        result = {}
        for name, values in partial.items():
            result[name] = np.full(len(history), np.nan)
            result[name][valid] = values
        return result
    a = fill_history(history)
    z = np.log1p(a)
    n, c = z.shape
    out = {}
    for k in (3, 6):
        common = np.empty(n)
        for cat in np.unique(codes):
            mask = codes == cat
            factor = np.median(z[mask], axis=0)
            growth = np.median((factor[12:] - factor[:-12])[-k:]) if c > 12 else 0.
            target = c + h - 1 - 12
            if not 0 <= target < c:
                raise ValueError('Seasonal target outside observed prefix')
            local = np.median(z[mask, -3:] - factor[-3:], axis=1)
            common[mask] = np.maximum(np.expm1(factor[target] + growth + local), 0)
        seasonal = np.array([baseline_forecasts(row, h)['seasonal_scaled'] for row in history])
        for weight in (.25, .75):
            out[f'panel_k{k}_w{weight:g}'] = weight * common + (1-weight) * seasonal
    return out

def add_candidates(frame, history, codes):
    w = frame.copy()
    for h in w.horizon.unique():
        ix = w.horizon.eq(h)
        for name, pred in panel_candidates(history, codes, int(h)).items():
            w.loc[ix, name] = pred[w.loc[ix, 'series_index'].to_numpy(int)]
    for k in (3, 6):
        w[f'v2_panel_k{k}'] = .5*w.v2 + .5*w[f'panel_k{k}_w0.75']
    return w

def select_policy(tune):
    policy = {}
    names = ['v2'] + [f'panel_k{k}_w{w:g}' for k in (3,6) for w in (.25,.75)] + ['v2_panel_k3','v2_panel_k6']
    for (h, cat), g in tune.groupby(['horizon','category']):
        err = {name: float(np.abs(g[name]-g.actual).mean()) for name in names}
        name = min(names, key=lambda x: err[x])
        if h == 12 or err[name] > .98*err['v2']: name = 'v2'
        policy[f'{h}|{cat}'] = {'model':name, 'tuning_mae':err, 'rule':'2% minimum; h12 fixed'}
    return policy

def selected(frame, policy):
    return np.array([r['v2'] if r['horizon']==12 else r[policy[f"{r['horizon']}|{r['category']}"]['model']] for r in frame.to_dict('records')])

def signature():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('upgrade_v3.py','V3_PROTOCOL.md','data/consumption.parquet','upgrade_results/policy.json')}

def forecasting():
    cfg = json.loads((ROOT/'final_config.json').read_text())
    _, panel, dates, cats, _, training, _, _ = prepare(cfg)
    codes = np.array([cats.index(c) for _, c in panel.index])
    tcodes = np.array([cats.index(c) for _, c in training.index])
    old = json.loads((ROOT/'upgrade_results/protocol.json').read_text())
    policy2 = json.loads((ROOT/'upgrade_results/policy.json').read_text())
    pilot = json.loads((ROOT/'results/results.json').read_text())['pilot_territory_ids']
    groups = {int(t):name for name, ids in [('tuning',old['tuning_ids']),('calibration',old['calibration_ids']),('v2_test',old['test_ids']),('pilot',pilot)] for t in ids}
    sig = signature()
    if (OUT/'signature.json').exists() and json.loads((OUT/'signature.json').read_text()) != sig:
        raise RuntimeError('Code/protocol changed after run; use a new results directory')
    write_json(OUT/'signature.json',sig)
    # Freeze using only saved tuning V2 forecasts and prefix-only new predictions.
    tune = pd.read_parquet(ROOT/'upgrade_results/tuning_predictions.parquet')
    tune['v2'] = apply_policy(tune,policy2)
    lookup = {key:i for i,key in enumerate(panel.index)}
    tune['series_index'] = [lookup[(r.territory_id,r.category)] for r in tune.itertuples()]
    tune = pd.concat([add_candidates(g,panel.iloc[:,:c].to_numpy(),codes) for c,g in tune.groupby('origin')])
    policy = select_policy(tune)
    write_json(OUT/'policy.json',policy)
    tune.to_parquet(OUT/'tuning_candidates.parquet',index=False)
    print('V3 policy frozen', {k:v['model'] for k,v in policy.items()},flush=True)
    pipe = None
    frames = []
    for c in (12,18,21,23):
        cache = OUT/f'origin_{c}.parquet'
        if cache.exists(): frames.append(pd.read_parquet(cache)); continue
        hs = [12] if c == 12 else [h for h in (1,3,6) if c+h<=24]
        history = panel.iloc[:,:c].to_numpy()
        valid = np.isfinite(history).any(axis=1)
        sample = panel.loc[valid]
        history = history[valid]
        sample_codes = codes[valid]
        print(f'origin {c}: {len(history)} series',flush=True)
        if c != 12:
            if pipe is None: pipe = foundation_pipeline(cfg)
            adapt = adapted_foundation(pipe,history,max(hs),cfg)
            ridge = PooledRidge(cfg,cats).fit(training.iloc[:,:c].to_numpy(),tcodes,dates)
        parts = []
        for h in hs:
            base = [baseline_forecasts(row,h) for row in history]
            w = pd.DataFrame({'territory_id':sample.index.get_level_values(0),'category':sample.index.get_level_values(1),
                'series_index':np.arange(len(sample)),'origin':c,'horizon':h,'actual':sample.iloc[:,c+h-1].to_numpy(),
                'history_scale':np.maximum(np.nanmean(history,axis=1),1),
                'seasonal_scaled':[b['seasonal_scaled'] for b in base], 'damped_trend':[b['damped_trend'] for b in base]})
            w['pooled_ridge'] = w.damped_trend if c==12 else ridge.predict(history,sample_codes,dates,h)
            w['chronos_adapted'] = w.damped_trend if c==12 else adapt[:,h-1]
            w = candidates(w); w['v2'] = apply_policy(w,policy2)
            w = add_candidates(w,history,sample_codes)
            w['v3'] = selected(w,policy)
            w['cohort'] = w.territory_id.map(groups).fillna('remaining')
            parts.append(w[np.isfinite(w.actual)].copy())
        frame = pd.concat(parts,ignore_index=True)
        frame.to_parquet(cache,index=False); frames.append(frame)
        print(f'origin {c}: saved {len(frame)} pairs',flush=True)
    full = pd.concat(frames,ignore_index=True)
    calibration = full[full.cohort.eq('calibration')].copy()
    calibration['score'] = abs(calibration.v3-calibration.actual)/calibration.history_scale
    widths = {f'{h}|{cat}':quantile(g.groupby('territory_id').score.max()) for (h,cat),g in calibration.groupby(['horizon','category'])}
    write_json(OUT/'interval_calibration.json',widths)
    width = np.array([widths[f'{r.horizon}|{r.category}']*r.history_scale for r in full.itertuples()])
    full['lower'] = np.maximum(full.v3-width,0); full['upper'] = full.v3+width
    full.to_parquet(OUT/'predictions.parquet',index=False)
    names = ['v2','v3','seasonal_scaled','damped_trend']+[f'panel_k{k}_w{w:g}' for k in (3,6) for w in (.25,.75)]+['v2_panel_k3','v2_panel_k6']
    long = full.melt(id_vars=KEY+['actual','history_scale','cohort'],value_vars=names,var_name='model',value_name='predicted')
    for filename, by in [('metrics',['model','horizon']),('category_metrics',['model','horizon','category']),('cohort_metrics',['cohort','model','horizon'])]:
        metric_table(long,by).to_csv(OUT/f'{filename}.csv',index=False)
    oldtest = pd.read_parquet(ROOT/'upgrade_results/test_predictions.parquet')
    paired = full.merge(oldtest[KEY+['v2','prophet_default']],on=KEY,suffixes=('','_saved'),validate='one_to_one')
    if not np.allclose(paired.v2,paired.v2_saved,rtol=1e-5,atol=.05): raise AssertionError('V2 reproduction failed')
    paired.to_parquet(OUT/'paired_prophet.parquet',index=False)
    write_json(OUT/'completion.json',{'signature':sig,'pairs':len(full),'territories':int(full.territory_id.nunique()),'paired_prophet':len(paired),'v2_max_reproduction_difference':float(abs(paired.v2-paired.v2_saved).max())})
    print('Full panel forecasting complete',flush=True)

def factor_scores(y, factor, spread):
    growth = np.log1p(y[12:])-np.log1p(y[:-12])
    return np.r_[np.full(12,np.nan),(growth-factor)/spread]

def panel_alarms(z, window, threshold):
    alarms=[]; last=-100
    for t in range(12,len(z)):
        values=z[max(12,t-window+1):t+1]
        score=np.mean(values) if len(values)==window else 0.
        if abs(score)>=threshold and t-last>3: alarms.append(t); last=t
    return alarms

def detector_cases(a, ids, factor, spread, seed):
    rng=np.random.default_rng(seed); cases=[]
    for i in ids:
        for kind in ('control','outlier','up10','down10','up20','down20','slope'):
            y=a[i].copy(); start=int(rng.integers(15,20))
            if kind=='outlier':y[start]*=1.3
            elif kind=='slope':y[start:]*=np.exp(.03*np.arange(1,len(y)-start+1))
            elif kind!='control':y[start:]*=(1+(1 if kind.startswith('up') else -1)*int(kind[-2:])/100)
            cases.append({'series':int(i),'kind':kind,'change':None if kind in ('control','outlier') else start,
                          'z':innovations(y),'panel_z':factor_scores(y,factor,spread)})
    return cases

def detector_score(cases,method,threshold):
    tp=fp=fn=controls=0; delays=[]; details=[]
    for case in cases:
        if method.startswith('panel'): alarms=panel_alarms(case['panel_z'],int(method[-1]),threshold)
        else: alarms=[r['t'] for r in robust_chart(case['z'],method,threshold) if r['alarm'] and r['t']>=12]
        change=case['change']; hits=[] if change is None else [t for t in alarms if change<=t<=change+2]
        hit=bool(hits);tp+=hit;fp+=len(alarms)-hit;fn+=change is not None and not hit
        if hit:delays.append(hits[0]-change)
        controls+=case['kind']=='control' and bool(alarms)
        details.append({'series':case['series'],'kind':case['kind'],'change':change,'alarms':alarms})
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return {'method':method,'threshold':threshold,'tp':int(tp),'fp':int(fp),'fn':int(fn),'precision':p,'recall':r,
        'f1':2*p*r/max(p+r,1e-12),'control_rate':controls/sum(c['kind']=='control' for c in cases),
        'median_delay':float(np.median(delays)) if delays else None},details

def detectors():
    cfg=json.loads((ROOT/'final_config.json').read_text()); _,panel,_,_,_,_,_,_=prepare(cfg)
    sub=panel.xs('Все категории',level=1).dropna();a=sub.to_numpy()
    growth=np.log1p(a[:,12:])-np.log1p(a[:,:-12]);factor=np.median(growth,axis=0)
    spread=np.maximum(1.4826*np.median(abs(growth-factor),axis=0),.02)
    order=sorted(range(len(sub)),key=lambda i:hashlib.sha256(f'20261011:{sub.index[i]}'.encode()).hexdigest())
    cal=detector_cases(a,order[:300],factor,spread,20261012)
    test=detector_cases(a,order[300:600],factor,spread,20261013)
    rows=[];choices={};details=[]
    for method in ('cusum_v1','multiscale','panel1','panel2','panel3'):
        thresholds=[8] if method=='cusum_v1' else [5] if method=='multiscale' else [2,3,4,5,6,8]
        scores=[detector_score(cal,method,t)[0] for t in thresholds]
        rows.extend({**r,'split':'calibration'} for r in scores)
        eligible=[r for r in scores if r['control_rate']<=.05]
        best=min(eligible,key=lambda r:(-r['f1'],r['threshold'])) if eligible else min(scores,key=lambda r:(r['control_rate'],-r['f1']))
        choices[method]=best['threshold']
    write_json(OUT/'detector_policy.json',{'thresholds':choices,'calibration_ids':[int(sub.index[i]) for i in order[:300]],'test_ids':[int(sub.index[i]) for i in order[300:600]]})
    for method,t in choices.items():
        result,d=detector_score(test,method,t);rows.append({**result,'split':'test'});details.append({'method':method,'cases':d})
    pd.DataFrame(rows).to_csv(OUT/'detector_metrics.csv',index=False)
    write_json(OUT/'detector_cases.json',details)
    print('Semisynthetic detector evaluation complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['forecast','detector'],required=True)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    forecasting() if args.phase=='forecast' else detectors()
