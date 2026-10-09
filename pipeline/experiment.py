"""Reproduce paired forecasts, online detector evaluation and deployment exports.

Run from this directory: python experiment.py --phase all
All evaluation objectives and split rules are in PROTOCOL.md.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from benchmark import BASE_MODELS, baseline_forecasts, load_data
from forecasting import PooledRidge, fill_history, news_features, ranked_ids
from detection import detect, evaluate_detector, innovations, synthetic_cases

ROOT=Path(__file__).resolve().parent


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prophet_job(job):
    """Local fit, deterministic seed. Exceptions propagate: no silent dropped cases."""
    from prophet import Prophet
    logging.getLogger("cmdstanpy").disabled=True
    logging.getLogger("prophet").disabled=True
    dates,y,horizons,seed=job
    training=pd.DataFrame({"ds":dates,"y":y}).dropna()
    future=pd.DataFrame({"ds":[dates[-1]+pd.DateOffset(months=h) for h in horizons]})
    result={}
    for name,annual in [("prophet_default","auto"),("prophet_annual3",False)]:
        model=Prophet(yearly_seasonality=annual,weekly_seasonality=False,
                      daily_seasonality=False,uncertainty_samples=0,
                      changepoint_prior_scale=0.05,seasonality_prior_scale=10.0)
        if annual is False: model.add_seasonality("annual",period=365.25,fourier_order=3)
        model.fit(training,seed=seed)
        pred=model.predict(future).yhat.to_numpy()
        if not np.isfinite(pred).all(): raise ValueError("Invalid Prophet output")
        result[name]=np.maximum(pred,0).tolist()
    return result


def foundation_pipeline(config):
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(config["torch_threads"])
    return BaseChronosPipeline.from_pretrained(config["foundation_model"],
            revision=config["foundation_revision"],device_map="cpu",torch_dtype=torch.float32)


def foundation_predict(pipeline,history,horizon,batch_size):
    import torch
    values=fill_history(history)
    out=[]
    for i in range(0,len(values),batch_size):
        q,median=pipeline.predict_quantiles(torch.tensor(values[i:i+batch_size],dtype=torch.float32),
                                            prediction_length=horizon,quantile_levels=[0.5])
        # Use the 0.5 quantile, not the separate mean output.
        out.append(q[:,:,0].detach().cpu().numpy())
    pred=np.maximum(np.concatenate(out),0)
    if not np.isfinite(pred).all(): raise ValueError("Invalid foundation output")
    return pred


def prepare(config):
    df,panel,dates,_=load_data(ROOT/"data/consumption.parquet")
    pilot=json.loads((ROOT/"results/results.json").read_text())["pilot_territory_ids"]
    ids=ranked_ids(panel,config["seed"],pilot)[:config["evaluation_territories"]]
    train_ids=ranked_ids(panel,config["seed"]+1)[:config["training_territories"]]
    evaluation=panel.loc[panel.index.get_level_values(0).isin(ids)]
    training=panel.loc[panel.index.get_level_values(0).isin(train_ids)]
    cats=sorted(df.category.unique())
    return df,panel,dates,cats,evaluation,training,ids,train_ids


def metric_table(predictions,groups):
    rows=[]
    for key,g in predictions.groupby(groups,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        y,p=g.actual.to_numpy(),g.predicted.to_numpy()
        error=np.abs(p-y)
        denom=np.sum((y-y.mean())**2)
        rows.append({**dict(zip(groups,key)),"n":len(g),"mae":float(error.mean()),
                     "wape":float(error.sum()/np.abs(y).sum()),
                     "r2_pooled":float(1-np.sum((p-y)**2)/denom) if denom>0 else None,
                     "mean_scaled_absolute_error":float((error/g.history_scale).mean())})
    return pd.DataFrame(rows)


def forecast_experiment(config,out):
    started=time.perf_counter()
    df,panel,dates,cats,evaluation,training,ids,train_ids=prepare(config)
    events=pd.read_csv(ROOT/"data/news_events.csv")
    write_json(out/"cohort.json",{"evaluation_ids":ids,"training_ids":train_ids,
               "selection":"first-year coverage only, then SHA256 rank",
               "evaluation_series":len(evaluation),"categories":cats})
    signature={"config":sha(ROOT/"final_config.json"),"data":sha(ROOT/"data/consumption.parquet"),
               "news":sha(ROOT/"data/news_events.csv"),"forecasting_code":sha(ROOT/"forecasting.py"),
               "experiment_code":sha(ROOT/"experiment.py")}
    signature_path=out/"run_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text())!=signature:
        raise RuntimeError("Inputs/code changed; choose a fresh --output directory")
    write_json(signature_path,signature)
    codes=np.array([cats.index(c) for _,c in evaluation.index])
    train_codes=np.array([cats.index(c) for _,c in training.index])
    pipeline=foundation_pipeline(config)
    frames=[]
    runtime=[]
    for c in config["origins"]:
        cache=out/f"origin_{c}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache));continue
        start=time.perf_counter()
        horizons=[h for h in config["horizons"] if c+h<=len(dates)]
        history=evaluation.iloc[:,:c].to_numpy()
        train=training.iloc[:,:c].to_numpy()
        print(f"origin {c}: fitting pooled models",flush=True)
        ridge=PooledRidge(config,cats).fit(train,train_codes,dates)
        news=PooledRidge(config,cats,events).fit(train,train_codes,dates)
        predictions={}
        for h in horizons:
            predictions[("pooled_ridge",h)]=ridge.predict(history,codes,dates,h)
            predictions[("pooled_ridge_news",h)]=news.predict(history,codes,dates,h)
            simple=[baseline_forecasts(row,h) for row in history]
            for name in BASE_MODELS: predictions[(name,h)]=np.array([r[name] for r in simple])
        fp=foundation_predict(pipeline,history,max(horizons),config["foundation_batch_size"])
        for h in horizons: predictions[("chronos_bolt_tiny",h)]=fp[:,h-1]
        print(f"origin {c}: Prophet for {len(history)} series",flush=True)
        jobs=[(dates[:c],row,horizons,config["seed"]) for row in history]
        with ProcessPoolExecutor(max_workers=config["prophet_workers"]) as executor:
            fitted=[]
            for i,r in enumerate(executor.map(prophet_job,jobs,chunksize=1),1):
                fitted.append(r)
                if i%60==0: print(f"origin {c}: Prophet {i}/{len(history)}",flush=True)
        for name in ["prophet_default","prophet_annual3"]:
            for j,h in enumerate(horizons): predictions[(name,h)]=np.array([r[name][j] for r in fitted])
        records=[]
        scales=np.maximum(np.nanmean(np.abs(history),axis=1),1)
        for (name,h),pred in predictions.items():
            target=c+h-1
            split=("diagnostic_h12" if h==12 else
                   "validation" if target<=config["validation_last_target_index"] else
                   "test" if c>=config["test_first_origin"] else "bridge")
            for i,(territory,category) in enumerate(evaluation.index):
                actual=evaluation.iloc[i,target]
                if not np.isfinite(actual): continue
                records.append({"territory_id":int(territory),"category":category,
                    "origin":c,"as_of":str((dates[c-1]+pd.offsets.MonthBegin()).date()),
                    "horizon":h,"target":str(dates[target].date()),"split":split,
                    "actual":float(actual),"predicted":float(pred[i]),"model":name,
                    "history_scale":float(scales[i])})
        frame=pd.DataFrame(records)
        frame.to_parquet(cache,index=False)
        frames.append(frame)
        runtime.append({"origin":c,"seconds":time.perf_counter()-start,
                        "pooled_training_rows":ridge.training_rows})
        print(f"origin {c}: saved {len(frame)} paired forecasts",flush=True)
    predictions=pd.concat(frames,ignore_index=True)
    predictions.to_parquet(out/"paired_predictions.parquet",index=False)
    metrics=metric_table(predictions,["split","model","horizon"])
    metrics.to_csv(out/"forecast_metrics.csv",index=False)
    bycat=metric_table(predictions,["split","model","horizon","category"])
    bycat.to_csv(out/"category_metrics.csv",index=False)
    val=metrics[metrics.split.eq("validation")]
    selection={str(h):str(g.sort_values(["mae","model"]).iloc[0].model) for h,g in val.groupby("horizon")}
    selection["12"]=config["h12_deployment_model"]
    write_json(out/"selection.json",{"by_horizon":selection,"criterion":"validation MAE only for h=1,3,6",
                                     "h12":"fixed in protocol; not selected on diagnostic results"})
    selected=pd.concat([predictions[predictions.horizon.eq(int(h)) & predictions.model.eq(m)]
                        for h,m in selection.items()])
    selected.to_parquet(out/"selected_predictions.parquet",index=False)
    write_json(out/"forecast_runtime.json",{"wall_seconds":time.perf_counter()-started,"origins":runtime,
                 "foundation_model":config["foundation_model"],"foundation_revision":config["foundation_revision"],
                 "device":"cpu","versions":{n:importlib.metadata.version(n) for n in
                    ["numpy","pandas","scikit-learn","prophet","torch","transformers","chronos-forecasting"]}})
    print("forecast phase complete",flush=True)


def detector_experiment(config,out):
    results=[]; choices={}
    calibration=synthetic_cases(config["synthetic_calibration_seed"],config["synthetic_cases_per_kind"])
    for c in calibration: c["z"]=innovations(c["y"])
    for method,thresholds in config["detector_thresholds"].items():
        candidates=[]
        for threshold in thresholds:
            r,_=evaluate_detector(calibration,method,threshold,config["alarm_cooldown"],config["detection_tolerance"])
            r["split"]="calibration";r["length"]=60;results.append(r);candidates.append(r)
        choices[method]=max(candidates,key=lambda x:(x["f1"],-x["null_series_alarm_rate"],x["threshold"]))["threshold"]
    best=max([r for r in results if r["threshold"]==choices[r["method"]]],
             key=lambda x:(x["f1"],-x["null_series_alarm_rate"],x["method"]))["method"]
    write_json(out/"detector_selection.json",{"method":best,"thresholds":choices,
                 "selection":"maximum calibration F1, then lower null alarm rate; test not used"})
    detailed=[]
    for length in (60,24):
        cases=synthetic_cases(config["synthetic_test_seed"]+length,config["synthetic_cases_per_kind"],length)
        for c in cases: c["z"]=innovations(c["y"])
        for method,threshold in choices.items():
            r,rows=evaluate_detector(cases,method,threshold,config["alarm_cooldown"],config["detection_tolerance"])
            r.update(split="test",length=length);results.append(r)
            detailed.extend([{**row,"method":method,"length":length} for row in rows])
    pd.DataFrame(results).to_csv(out/"detector_metrics.csv",index=False)
    write_json(out/"synthetic_case_results.json",detailed)
    print(f"detector phase complete; selected {best}",flush=True)


def deploy(config,out):
    _,panel,dates,cats,evaluation,training,ids,train_ids=prepare(config)
    selection=json.loads((out/"selection.json").read_text())["by_horizon"]
    detector=json.loads((out/"detector_selection.json").read_text())
    events=pd.read_csv(ROOT/"data/news_events.csv")
    train_codes=np.array([cats.index(c) for _,c in training.index])
    codes=np.array([cats.index(c) for _,c in panel.index])
    history=panel.to_numpy()
    models={}
    for name in set(selection.values()):
        if name.startswith("pooled_ridge"):
            models[name]=PooledRidge(config,cats,events if name.endswith("news") else None).fit(training.to_numpy(),train_codes,dates)
    foundation=foundation_pipeline(config) if "chronos_bolt_tiny" in selection.values() else None
    foundation_output=foundation_predict(foundation,history,12,config["foundation_batch_size"]) if foundation else None
    # If a Prophet variant wins validation, compute it for every production series.
    ph={}
    ph_h=[int(h) for h,m in selection.items() if m.startswith("prophet")]
    if ph_h:
        with ProcessPoolExecutor(max_workers=config["prophet_workers"]) as executor:
            for i,result in enumerate(executor.map(prophet_job,[(dates,row,ph_h,config["seed"]) for row in history],chunksize=1)):
                ph[i]=result
                if i%1000==0: print(f"deployment Prophet: {i}/{len(history)}",flush=True)
    records=[]
    for h in config["horizons"]:
        name=selection[str(h)]
        if name in models: pred=models[name].predict(history,codes,dates,h)
        elif name=="chronos_bolt_tiny": pred=foundation_output[:,h-1]
        elif name.startswith("prophet"): pred=np.array([ph[i][name][ph_h.index(h)] for i in range(len(history))])
        else: pred=np.array([baseline_forecasts(row,h)[name] for row in history])
        for i,(territory,category) in enumerate(panel.index):
            records.append({"territory_id":int(territory),"category":category,"horizon":h,
                "data_through":str(dates[-1].date()),"target":str((dates[-1]+pd.DateOffset(months=h)).date()),
                "predicted":float(pred[i]),"model":name,"status":"historical_export_not_current_2026_forecast"})
    pd.DataFrame(records).to_csv(out/"forecast_export.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    alarms=[];latest=[]
    for (territory,category),row in panel.iterrows():
        scores=detect(row.to_numpy(),detector["method"],detector["thresholds"][detector["method"]],config["alarm_cooldown"])
        for r in scores:
            if not (r["alarm"] or r["warning"]): continue
            as_of=dates[r["t"]]+pd.offsets.MonthBegin()
            known=events[(pd.to_datetime(events.available_at)<as_of) &
                         (pd.to_datetime(events.available_at)>=as_of-pd.Timedelta(days=90)) & events.delta_pp.ne(0)]
            alarms.append({"territory_id":int(territory),"category":category,"observation_month":str(dates[r["t"]].date()),
                "as_of":str(as_of.date()),"method":detector["method"],"score":r["score"],
                "alarm":r["alarm"],"warning":r["warning"],"direction":r["direction"],
                "known_news_ids":";".join(known.event_id),"causality":"not_established"})
        r=scores[-1]
        latest.append({"territory_id":int(territory),"category":category,"score":r["score"],
                       "alarm":r["alarm"],"warning":r["warning"],"direction":r["direction"]})
    pd.DataFrame(alarms).to_csv(out/"signals.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    pd.DataFrame(latest).to_csv(out/"latest_signals.csv",index=False)
    print(f"exports: {len(records)} forecasts, {len(alarms)} warning/alarm observations",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["all","forecast","detector","export"],default="all")
    p.add_argument("--output",type=Path,default=ROOT/"final_results")
    args=p.parse_args()
    config=json.loads((ROOT/"final_config.json").read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase in ("all","forecast"): forecast_experiment(config,args.output)
    if args.phase in ("all","detector"): detector_experiment(config,args.output)
    if args.phase in ("all","export"): deploy(config,args.output)


if __name__=="__main__":
    main()
