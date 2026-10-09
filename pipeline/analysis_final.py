"""Paired uncertainty summaries and a publication-delay sensitivity experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import baseline_forecasts
from experiment import (ROOT,prepare,prophet_job,foundation_pipeline,foundation_predict,
                        metric_table,write_json)
from forecasting import PooledRidge


def paired_comparisons(out,seed):
    p=pd.read_parquet(out/"paired_predictions.parquet")
    selected=pd.read_parquet(out/"selected_predictions.parquet")
    rng=np.random.default_rng(seed)
    results=[]
    for h,g in selected[selected.split.eq("test")].groupby("horizon"):
        base=p[p.split.eq("test")&p.horizon.eq(h)&p.model.eq("prophet_default")]
        merged=g.merge(base,on=["territory_id","category","origin","horizon","target"],suffixes=("_selected","_base"),validate="one_to_one")
        merged["selected_error"]=abs(merged.actual_selected-merged.predicted_selected)
        merged["base_error"]=abs(merged.actual_base-merged.predicted_base)
        grouped=merged.groupby("territory_id")[["selected_error","base_error"]].agg(["sum","count"])
        a=grouped.to_numpy()
        samples=rng.integers(0,len(a),(2000,len(a)))
        draw=a[samples].sum(axis=1)
        difference=draw[:,0]/draw[:,1]-draw[:,2]/draw[:,3]
        results.append({"horizon":int(h),"model":str(g.model.iloc[0]),"n":len(merged),
            "mae_selected":float(merged.selected_error.mean()),"mae_prophet":float(merged.base_error.mean()),
            "relative_mae_change_pct":float((merged.selected_error.mean()/merged.base_error.mean()-1)*100),
            "mae_difference_ci_low":float(np.quantile(difference,.025)),
            "mae_difference_ci_high":float(np.quantile(difference,.975)),
            "territory_win_share":float((a[:,0]/a[:,1]<a[:,2]/a[:,3]).mean()),
            "bootstrap":"2000 paired municipality-cluster draws; shared calendar dependence remains"})
    write_json(out/"paired_comparison.json",results)
    news=[]
    m=pd.read_csv(out/"forecast_metrics.csv")
    for split in ["validation","test","diagnostic_h12"]:
        for h in (1,3,6,12):
            g=m[m.split.eq(split)&m.horizon.eq(h)].set_index("model")
            if "pooled_ridge" not in g.index:continue
            a=float(g.loc["pooled_ridge","mae"]);b=float(g.loc["pooled_ridge_news","mae"])
            news.append({"split":split,"horizon":h,"without_news_mae":a,"with_news_mae":b,
                         "relative_change_pct":(b/a-1)*100})
    pd.DataFrame(news).to_csv(out/"news_ablation.csv",index=False)


def delay_sensitivity(config,out):
    _,panel,dates,cats,evaluation,training,ids,train_ids=prepare(config)
    choice=json.loads((out/"selection.json").read_text())["by_horizon"]
    events=pd.read_csv(ROOT/"data/news_events.csv")
    codes=np.array([cats.index(c) for _,c in evaluation.index])
    tcodes=np.array([cats.index(c) for _,c in training.index])
    pipe=foundation_pipeline(config) if "chronos_bolt_tiny" in choice.values() else None
    rows=[]
    for origin in [c for c in config["origins"] if c>=18]:
        c=origin-1
        history=evaluation.iloc[:,:c].to_numpy()
        horizons=[h for h in (1,3,6) if origin+h<=len(dates)]
        models={}
        for name in set(choice[str(h)] for h in horizons):
            if name.startswith("pooled_ridge"):
                models[name]=PooledRidge(config,cats,events if name.endswith("news") else None).fit(training.iloc[:,:c].to_numpy(),tcodes,dates)
        # Same target dates, one fewer observed month, effective h+1.
        if pipe is not None: fp=foundation_predict(pipe,history,max(horizons)+1,config["foundation_batch_size"])
        ph_h=[h+1 for h in horizons if choice[str(h)].startswith("prophet")]
        ph=[]
        if ph_h:
            with ProcessPoolExecutor(max_workers=config["prophet_workers"]) as executor:
                ph=list(executor.map(prophet_job,[(dates[:c],y,ph_h,config["seed"]) for y in history]))
        for h in horizons:
            name=choice[str(h)]
            if name in models:pred=models[name].predict(history,codes,dates,h+1)
            elif name=="chronos_bolt_tiny":pred=fp[:,h]
            elif name.startswith("prophet"):pred=np.array([r[name][ph_h.index(h+1)] for r in ph])
            else:pred=np.array([baseline_forecasts(y,h+1)[name] for y in history])
            for i,(territory,category) in enumerate(evaluation.index):
                actual=evaluation.iloc[i,origin+h-1]
                if not np.isfinite(actual):continue
                rows.append({"territory_id":int(territory),"category":category,"origin":origin,"horizon":h,
                             "actual":float(actual),"predicted":float(pred[i]),"model":name,
                             "history_scale":float(max(np.nanmean(history[i]),1))})
        print(f"delay sensitivity origin {origin} complete",flush=True)
    frame=pd.DataFrame(rows)
    frame.to_parquet(out/"delay_predictions.parquet",index=False)
    metric_table(frame,["model","horizon"]).to_csv(out/"delay_metrics.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"final_results")
    parser.add_argument("--skip-delay",action="store_true")
    a=parser.parse_args();cfg=json.loads((ROOT/"final_config.json").read_text())
    paired_comparisons(a.output,cfg["seed"])
    if not a.skip_delay:delay_sensitivity(cfg,a.output)
