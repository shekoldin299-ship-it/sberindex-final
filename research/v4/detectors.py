"""Causal local-change detector; thresholds calibrated on separate territories."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'pipeline'))
from benchmark import load_data
from upgrade_v3 import detector_score, detector_cases


def change_scores(y, factor):
    # Factor at month t is observable with the panel at t; no later month is used.
    residual=np.log1p(y)-factor
    scores=np.zeros((2,len(y)))
    for t in range(12,len(y)):
        past=np.diff(residual[:t])
        scale=max(1.4826*np.median(abs(past-np.median(past))),.015)
        for w in (2,3):
            # Compare recent levels to an earlier robust local reference.
            left=residual[max(0,t-w-5):t-w+1]
            recent=residual[t-w+1:t+1]
            shift=recent-np.median(left)
            consistent=np.all(shift>0) or np.all(shift<0)
            scores[w-2,t]=(np.min(abs(shift))/scale if consistent else 0.)
    return scores


def alarms(scores, threshold):
    out=[]
    for t in range(12,len(scores)):
        if scores[t]>=threshold and (not out or t-out[-1]>3):out.append(t)
    return out


def score(cases, method, threshold):
    tp=fp=fn=controls=0;delays=[];details=[]
    for c in cases:
        a=alarms(c[method],threshold);change=c['change']
        hits=[] if change is None else [t for t in a if change<=t<=change+2]
        hit=bool(hits);tp+=hit;fp+=len(a)-hit;fn+=change is not None and not hit
        if hit:delays.append(hits[0]-change)
        controls+=c['kind']=='control' and bool(a)
        details.append({'series':c['series'],'kind':c['kind'],'change':change,'alarms':a})
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return {'method':method,'threshold':threshold,'tp':int(tp),'fp':int(fp),'fn':int(fn),
        'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-12),
        'control_rate':controls/sum(c['kind']=='control' for c in cases),
        'median_delay':float(np.median(delays)) if delays else None},details


def generate(a, ids, seed, factor):
    rng=np.random.default_rng(seed);out=[]
    for i in ids:
        for kind in ('control','outlier','up10','down10','up20','down20','slope'):
            y=a[i].copy();start=int(rng.integers(15,20))
            if kind=='outlier':y[start]*=1.3
            elif kind=='slope':y[start:]*=np.exp(.03*np.arange(1,len(y)-start+1))
            elif kind!='control':y[start:]*=1+(1 if kind.startswith('up') else -1)*int(kind[-2:])/100
            s=change_scores(y,factor)
            out.append({'series':int(i),'kind':kind,'change':None if kind in ('control','outlier') else start,
                'confirmed2':s[0],'confirmed3':s[1]})
    return out


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    _,panel,_,_=load_data(ROOT/'pipeline/data/consumption.parquet')
    panel=panel.xs('Все категории',level=1).dropna();a=panel.to_numpy();z=np.log1p(a)
    factor=np.median(z,axis=0)
    growth=z[:,12:]-z[:,:-12];gf=np.median(growth,axis=0)
    spread=np.maximum(1.4826*np.median(abs(growth-gf),axis=0),.02)
    order=sorted(range(len(panel)),key=lambda i:hashlib.sha256(f'20261011:{panel.index[i]}'.encode()).hexdigest())
    calibration=generate(a,order[:300],20261012,factor)
    rows=[];policies={}
    for method in ('confirmed2','confirmed3'):
        options=[score(calibration,method,t)[0] for t in (1.5,2,2.5,3,4,5,6,8)]
        rows.extend(dict(r,split='calibration') for r in options)
        eligible=[r for r in options if r['control_rate']<=.05]
        best=max(eligible,key=lambda r:(r['f1'],-r['control_rate'])) if eligible else min(options,key=lambda r:r['control_rate'])
        policies[method]=best['threshold']
    (out/'policy.json').write_text(json.dumps(policies,indent=2))
    test=generate(a,order[300:600],20261013,factor)
    details=[]
    for method,t in policies.items():
        r,d=score(test,method,t);rows.append(dict(r,split='test'));details.append({'method':method,'cases':d})
    # Exactly the previous reference cases, unchanged seeds, territory order and scoring.
    old=detector_cases(a,order[300:600],gf,spread,20261013)
    for method,t in [('cusum_v1',8),('multiscale',5)]:
        r,_=detector_score(old,method,t);rows.append(dict(r,split='test'))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'cases.json').write_text(json.dumps(details,ensure_ascii=False))
    print(pd.DataFrame(rows).query('split == "test"').to_string(index=False),flush=True)


if __name__=='__main__':main(sys.argv[1])
