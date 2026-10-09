"""One disclosed second design iteration: baseline until mature evidence supports a change."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd


def main(inputs,out):
    inputs=Path(inputs);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    past={};frames=[];policy=[]
    for c in range(12,24):
        w=pd.read_parquet(inputs/f'origin_{c}.parquet');w['safe_mix']=w.v3
        for cat,g in w.groupby('category'):
            history=past.get(cat,[])[-3:]
            # Baseline-preserving warm-up and 2% margin are fixed before this run.
            adopt=len(history)>=3 and np.mean([r[1] for r in history])<=.98*np.mean([r[0] for r in history])
            if adopt:w.loc[g.index,'safe_mix']=.5*g.v3+.5*g.online_mix
            policy.append({'origin':c,'category':cat,'weight_new':.5 if adopt else 0,
                           'latest_known_target':c-1,'n_mature_origins':len(history)})
        # Update after prediction; h=1 labels are available to the following origin.
        for cat,g in w.groupby('category'):
            past.setdefault(cat,[]).append((float(abs(g.v3-g.actual).mean()),float(abs(g.online_mix-g.actual).mean())))
        frames.append(w)
        w.to_parquet(out/f'origin_{c}.parquet',index=False)
    full=pd.concat(frames);rows=[]
    for c,g in full.groupby('origin'):
        rows.append({'origin':c,'n':len(g),**{m:float(abs(g[m]-g.actual).mean()) for m in ['v3','online_mix','safe_mix']}})
    pd.DataFrame(rows).to_csv(out/'origin_metrics.csv',index=False)
    result={m:float(abs(full[m]-full.actual).mean()) for m in ['v3','online_mix','safe_mix']}
    delta=np.array([r['safe_mix']-r['v3'] for r in rows]);rng=np.random.default_rng(20261009)
    result['month_bootstrap_ci']=np.quantile(delta[rng.integers(0,12,(10000,12))].mean(axis=1),[.025,.975]).tolist()
    result['status']='second_design_iteration_after_run01_not_independent_confirmation'
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    (out/'policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2))
    print(result,flush=True)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
