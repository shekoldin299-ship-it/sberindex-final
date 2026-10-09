"""Paired geographic bootstrap and predeclared-model ablation; no further selection."""
import json
import numpy as np
import pandas as pd
from experiment import write_json
from upgrade_v2 import OUT

def run():
    p=pd.read_parquet(OUT/'test_predictions.parquet');policy=json.loads((OUT/'policy.json').read_text())
    p['without_foundation']=p.v2
    for i,r in p.iterrows():
        if r.horizon!=12 and 'chronos' in policy[f'{int(r.horizon)}|{r.category}']['model']:p.loc[i,'without_foundation']=r.seasonal_scaled
    results=[];rng=np.random.default_rng(20261010)
    for h,g in p.groupby('horizon'):
        for other in ('v1','seasonal_scaled','prophet_default','without_foundation'):
            paired=g.assign(a=abs(g.v2-g.actual),b=abs(g[other]-g.actual)).groupby('territory_id')[['a','b']].agg(['sum','count']).to_numpy()
            sums=paired[rng.integers(0,len(paired),(2000,len(paired)))].sum(axis=1);diff=sums[:,0]/sums[:,1]-sums[:,2]/sums[:,3]
            mae=float(abs(g.v2-g.actual).mean());baseline=float(abs(g[other]-g.actual).mean())
            results.append({'horizon':int(h),'comparator':other,'v2_mae':mae,'comparator_mae':baseline,'change_pct':(mae/baseline-1)*100,'difference_ci95':[float(np.quantile(diff,.025)),float(np.quantile(diff,.975))],'territories':len(paired)})
    write_json(OUT/'paired_comparisons.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':run()
