"""Retrospective V3-policy comparison across all monthly origins."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from experiment import prepare,foundation_pipeline
from forecasting import PooledRidge
from benchmark import baseline_forecasts
from upgrade_v2 import adapted_foundation,candidates,apply_policy
from upgrade_v3 import add_candidates,selected


def main(inputs,out):
    inputs=Path(inputs);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'pipeline/final_config.json').read_text())
    _,panel,dates,cats,_,training,_,_=prepare(cfg)
    p2=json.loads((ROOT/'pipeline/upgrade_results/policy.json').read_text())
    p3=json.loads((ROOT/'pipeline/v3_results/policy.json').read_text())
    tcodes=np.array([cats.index(c) for _,c in training.index]);pipe=foundation_pipeline(cfg)
    saved=pd.read_parquet(ROOT/'pipeline/v3_results/predictions.parquet')
    frames=[];start=time.monotonic()
    for c in range(12,24):
        w=pd.read_parquet(inputs/f'origin_{c}.parquet').query('horizon==1').copy()
        if c in (18,21,23):
            old=saved.query('origin==@c and horizon==1')
        else:
            valid=panel.iloc[:,:c].notna().any(axis=1).to_numpy();sample=panel.loc[valid]
            history=sample.iloc[:,:c].to_numpy();codes=np.array([cats.index(k) for _,k in sample.index])
            ridge=PooledRidge(cfg,cats).fit(training.iloc[:,:c].to_numpy(),tcodes,dates)
            adapt=adapted_foundation(pipe,history,1,cfg)
            bases=[baseline_forecasts(y,1) for y in history]
            old=pd.DataFrame({'territory_id':sample.index.get_level_values(0),'category':sample.index.get_level_values(1),
                'series_index':np.arange(len(sample)),'origin':c,'horizon':1,
                'pooled_ridge':ridge.predict(history,codes,dates,1),'chronos_adapted':adapt[:,0],
                'seasonal_scaled':[b['seasonal_scaled'] for b in bases],'damped_trend':[b['damped_trend'] for b in bases]})
            old=candidates(old);old['v2']=apply_policy(old,p2)
            old=add_candidates(old,history,codes);old['v3']=selected(old,p3)
        w=w.merge(old[['territory_id','category','origin','horizon','v3']],
                  on=['territory_id','category','origin','horizon'],validate='one_to_one')
        frames.append(w);w.to_parquet(out/f'origin_{c}.parquet',index=False)
        print(f'expanded V3 origin={c} elapsed={time.monotonic()-start:.1f}s',flush=True)
    full=pd.concat(frames);rows=[]
    for c,g in full.groupby('origin'):
        rows.append({'origin':c,'n':len(g),'v3':float(abs(g.v3-g.actual).mean()),
                     'online_mix':float(abs(g.online_mix-g.actual).mean())})
    pd.DataFrame(rows).to_csv(out/'origin_metrics.csv',index=False)
    diffs=np.array([r['online_mix']-r['v3'] for r in rows])
    rng=np.random.default_rng(20261009)
    samples=diffs[rng.integers(0,len(diffs),(10000,len(diffs)))].mean(axis=1)
    result={'n':len(full),'v3_mae':float(abs(full.v3-full.actual).mean()),
        'online_mix_mae':float(abs(full.online_mix-full.actual).mean()),
        'month_bootstrap_ci':np.quantile(samples,[.025,.975]).tolist(),
        'bootstrap_caution':'12 dependent calendar months; diagnostic only',
        'baseline_caution':'V3 policy retrospectively fixed; not historically available at every origin'}
    (out/'comparison.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
