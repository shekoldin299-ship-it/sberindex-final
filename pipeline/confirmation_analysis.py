"""Summaries of the fixed candidate on additional territories; no selection."""
import json
import numpy as np
import pandas as pd
from experiment import ROOT,write_json


def run():
    out=ROOT/"final_results"
    p=pd.read_parquet(out/"confirmation_predictions.parquet")
    rng=np.random.default_rng(20261009)
    results=[]
    for h in (1,3,6,12):
        a=p[p.model.eq("release_candidate")&p.horizon.eq(h)]
        b=p[p.model.eq("prophet_default")&p.horizon.eq(h)]
        g=a.merge(b,on=["territory_id","category","origin","horizon","target"],suffixes=("_a","_b"),validate="one_to_one")
        g["ae_a"]=abs(g.actual_a-g.predicted_a);g["ae_b"]=abs(g.actual_b-g.predicted_b)
        v=g.groupby("territory_id")[["ae_a","ae_b"]].agg(["sum","count"]).to_numpy()
        sample=v[rng.integers(0,len(v),(2000,len(v)))].sum(axis=1)
        diff=sample[:,0]/sample[:,1]-sample[:,2]/sample[:,3]
        results.append({"horizon":h,"n":len(g),"territories_with_targets":len(v),
            "model":"pooled_ridge" if h!=12 else "damped_trend",
            "mae_candidate":float(g.ae_a.mean()),"mae_prophet":float(g.ae_b.mean()),
            "relative_mae_change_pct":float((g.ae_a.mean()/g.ae_b.mean()-1)*100),
            "mae_difference_ci_low":float(np.quantile(diff,.025)),"mae_difference_ci_high":float(np.quantile(diff,.975)),
            "territory_win_share":float((v[:,0]/v[:,1]<v[:,2]/v[:,3]).mean()),
            "interpretation":"additional territories, shared calendar; h12 is single-origin diagnostic"})
    write_json(out/"confirmation_comparison.json",results)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=="__main__":run()
