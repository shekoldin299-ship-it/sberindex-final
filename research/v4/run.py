"""Run from repository root: python research/v4/run.py --output /new/directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from models import PanelModel, MEMBERS, mixture_weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from benchmark import load_data


def metrics(frame, names, groups):
    rows=[]
    for key,g in frame.groupby(groups):
        key=key if isinstance(key,tuple) else (key,)
        y=g.actual.to_numpy(); sst=((y-y.mean())**2).sum()
        for name in names:
            p=g[name].to_numpy(); e=p-y
            rows.append(dict(zip(groups,key))|{'model':name,'n':len(g),
                'mae':float(abs(e).mean()), 'wape':float(abs(e).sum()/abs(y).sum()),
                'r2':float(1-(e**2).sum()/sst) if sst else None})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    source=ROOT/'pipeline/data/consumption.parquet'
    _,panel,dates,_=load_data(source)
    codes=pd.Categorical(panel.index.get_level_values(1),categories=sorted(panel.index.get_level_values(1).unique())).codes
    signature={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [source,HERE/'models.py',HERE/'run.py',HERE/'PROTOCOL.md']}
    (out/'signature.json').write_text(json.dumps(signature,indent=2))
    complete=panel.notna().all(axis=1).to_numpy()
    frames=[]; losses=[]; policies=[]
    for c in range(12,24):
        valid=panel.iloc[:,:c].notna().any(axis=1).to_numpy()
        sample=panel.loc[valid];history=sample.iloc[:,:c].to_numpy()
        model=PanelModel(history,codes[valid])
        originframes=[]
        for h in (1,3,6,12):
            if c+h>24:continue
            pred=model.predict(h)
            w=pd.DataFrame({'territory_id':sample.index.get_level_values(0),
                'category':sample.index.get_level_values(1),'origin':c,'horizon':h,
                'target_index':c+h-1,'actual':sample.iloc[:,c+h-1].to_numpy(),
                'complete':complete[valid],**pred})
            w['online_mix']=np.nan
            for cat, g in w.groupby('category'):
                weights=mixture_weights(losses,cat,h,c)
                w.loc[g.index,'online_mix']=g[MEMBERS].to_numpy()@weights
                policies.append({'origin':c,'horizon':h,'category':cat,
                    'weights':dict(zip(MEMBERS,weights.tolist()))})
            w=w[np.isfinite(w.actual)].copy()
            for cat,g in w.groupby('category'):
                losses.append({'origin':c,'target_index':c+h-1,'horizon':h,'category':cat,
                    **{name:float(abs(g[name]-g.actual).mean()) for name in MEMBERS}})
            originframes.append(w)
        frame=pd.concat(originframes,ignore_index=True)
        frame.to_parquet(out/f'origin_{c}.parquet',index=False)
        frames.append(frame)
        print(f'origin={c} pairs={len(frame)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    full=pd.concat(frames,ignore_index=True)
    key=['territory_id','category','origin','horizon']
    assert not full.duplicated(key).any()
    names=['seasonal']+MEMBERS+['online_mix']
    assert np.isfinite(full[names].to_numpy()).all()
    metrics(full,names,['horizon']).to_csv(out/'metrics.csv',index=False)
    metrics(full,names,['horizon','category']).to_csv(out/'category_metrics.csv',index=False)
    metrics(full,names,['origin','horizon']).to_csv(out/'origin_metrics.csv',index=False)
    metrics(full[full.complete],names,['origin','horizon']).to_csv(out/'complete_origin_metrics.csv',index=False)
    (out/'online_policy.json').write_text(json.dumps(policies,ensure_ascii=False,indent=2))
    old=pd.read_parquet(ROOT/'pipeline/v3_results/predictions.parquet')[key+['actual','v3']]
    paired=full.merge(old,on=key,validate='one_to_one',suffixes=('','_old'))
    assert np.allclose(paired.actual,paired.actual_old,atol=1e-8)
    assert len(paired)==len(old)
    metrics(paired,names+['v3'],['horizon']).to_csv(out/'paired_v3.csv',index=False)
    metrics(paired[paired.complete],names+['v3'],['origin','horizon']).to_csv(out/'paired_complete_v3.csv',index=False)
    (out/'completion.json').write_text(json.dumps({'pairs':len(full),'paired_v3':len(paired),
        'runtime_seconds':time.monotonic()-start,'origins':12,'status':'research_not_independent_future_test'},indent=2))
    print(pd.read_csv(out/'paired_v3.csv').pivot(index='model',columns='horizon',values='mae').round(2).to_string(),flush=True)


if __name__=='__main__':main()
