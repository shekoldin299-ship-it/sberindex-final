"""Fixed post-development candidate; confirm on additional territories, then export."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from prophet import Prophet
from benchmark import baseline_forecasts,load_data
from forecasting import PooledRidge,ranked_ids
from detection import detect
from experiment import ROOT,prepare,foundation_pipeline,foundation_predict,metric_table,write_json


def prophet_only(job):
    dates,history,horizons,seed=job
    logging.getLogger("cmdstanpy").disabled=True
    logging.getLogger("prophet").disabled=True
    model=Prophet(yearly_seasonality="auto",weekly_seasonality=False,daily_seasonality=False,
                  uncertainty_samples=0,changepoint_prior_scale=.05,seasonality_prior_scale=10)
    model.fit(pd.DataFrame({"ds":dates,"y":history}).dropna(),seed=seed)
    future=pd.DataFrame({"ds":[dates[-1]+pd.DateOffset(months=h) for h in horizons]})
    return np.maximum(model.predict(future).yhat.to_numpy(),0)


def candidate_predict(ridge,history,codes,dates,horizon):
    if horizon==12:
        return np.array([baseline_forecasts(y,12)["damped_trend"] for y in history])
    if history.shape[1]<18:
        return np.array([baseline_forecasts(y,horizon)["seasonal_scaled"] for y in history])
    return ridge.predict(history,codes,dates,horizon)


def confirmation(cfg,rcfg,out):
    _,panel,dates,cats,evaluation,training,old_ids,train_ids=prepare(cfg)
    pilot=json.loads((ROOT/"results/results.json").read_text())["pilot_territory_ids"]
    ids=ranked_ids(panel,rcfg["seed"],pilot+old_ids)[:rcfg["confirmation_territories"]]
    sample=panel.loc[panel.index.get_level_values(0).isin(ids)]
    codes=np.array([cats.index(c) for _,c in sample.index])
    tcodes=np.array([cats.index(c) for _,c in training.index])
    signature={"config":hashlib.sha256((ROOT/"release_config.json").read_bytes()).hexdigest(),
               "code":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"ids":ids,
               "interpretation":rcfg["interpretation"]}
    write_json(out/"confirmation_protocol.json",signature)
    pipe=foundation_pipeline(cfg)
    rows=[]
    for c in rcfg["origins"]:
        horizons=[12] if c==12 else [h for h in (1,3,6) if c+h<=24]
        history=sample.iloc[:,:c].to_numpy()
        ridge=PooledRidge(cfg,cats).fit(training.iloc[:,:c].to_numpy(),tcodes,dates)
        chronos=foundation_predict(pipe,history,max(horizons),cfg["foundation_batch_size"])
        print(f"confirmation origin {c}: Prophet {len(history)} series",flush=True)
        with ProcessPoolExecutor(max_workers=cfg["prophet_workers"]) as executor:
            prophets=[]
            for i,p in enumerate(executor.map(prophet_only,[(dates[:c],row,horizons,cfg["seed"]) for row in history]),1):
                prophets.append(p)
                if i%120==0:print(f"confirmation {c}: {i}/{len(history)}",flush=True)
        prophets=np.array(prophets)
        for j,h in enumerate(horizons):
            preds={"release_candidate":candidate_predict(ridge,history,codes,dates,h),
                   "prophet_default":prophets[:,j],"chronos_bolt_tiny":chronos[:,h-1],
                   "seasonal_scaled":np.array([baseline_forecasts(row,h)["seasonal_scaled"] for row in history])}
            for model,values in preds.items():
                for i,(tid,cat) in enumerate(sample.index):
                    actual=sample.iloc[i,c+h-1]
                    if not np.isfinite(actual):continue
                    rows.append({"territory_id":int(tid),"category":cat,"origin":c,"horizon":h,
                        "target":str(dates[c+h-1].date()),"actual":float(actual),"predicted":float(values[i]),
                        "model":model,"history_scale":float(max(np.nanmean(history[i]),1)),
                        "split":"diagnostic_h12" if h==12 else "confirmation"})
    p=pd.DataFrame(rows)
    p.to_parquet(out/"confirmation_predictions.parquet",index=False)
    metric_table(p,["split","model","horizon"]).to_csv(out/"confirmation_metrics.csv",index=False)
    metric_table(p,["split","model","horizon","category"]).to_csv(out/"confirmation_category_metrics.csv",index=False)
    write_json(out/"release_selection.json",{"policy":rcfg["candidate"],"frozen_before_confirmation":True,
        "by_horizon":{"1":"pooled_ridge","3":"pooled_ridge","6":"pooled_ridge","12":"damped_trend"}})
    print("confirmation complete",flush=True)


def delay(cfg,out):
    _,panel,dates,cats,evaluation,training,ids,train_ids=prepare(cfg)
    ids=json.loads((out/"confirmation_protocol.json").read_text())["ids"]
    sample=panel.loc[panel.index.get_level_values(0).isin(ids)]
    codes=np.array([cats.index(c) for _,c in sample.index])
    tcodes=np.array([cats.index(c) for _,c in training.index])
    rows=[]
    for origin in (18,21,23):
        c=origin-1
        history=sample.iloc[:,:c].to_numpy()
        ridge=PooledRidge(cfg,cats).fit(training.iloc[:,:c].to_numpy(),tcodes,dates)
        for h in (1,3,6):
            if origin+h>24:continue
            # The release policy's warmup is based on observed history length.
            if c<18:pred=np.array([baseline_forecasts(y,h+1)["seasonal_scaled"] for y in history])
            else:pred=ridge.predict(history,codes,dates,h+1)
            for i,(tid,cat) in enumerate(sample.index):
                actual=sample.iloc[i,origin+h-1]
                if not np.isfinite(actual):continue
                rows.append({"territory_id":int(tid),"category":cat,"origin":origin,"horizon":h,
                    "actual":float(actual),"predicted":float(pred[i]),"model":"release_candidate",
                    "history_scale":float(max(np.nanmean(history[i]),1))})
    p=pd.DataFrame(rows);p.to_parquet(out/"release_delay_predictions.parquet",index=False)
    metric_table(p,["model","horizon"]).to_csv(out/"release_delay_metrics.csv",index=False)


def export(cfg,out):
    _,panel,dates,cats,_,training,_,_=prepare(cfg)
    codes=np.array([cats.index(c) for _,c in panel.index])
    tcodes=np.array([cats.index(c) for _,c in training.index])
    ridge=PooledRidge(cfg,cats).fit(training.to_numpy(),tcodes,dates)
    history=panel.to_numpy();records=[]
    for h in cfg["horizons"]:
        pred=candidate_predict(ridge,history,codes,dates,h)
        for i,(tid,cat) in enumerate(panel.index):
            records.append({"territory_id":int(tid),"category":cat,"horizon":h,
                "data_through":str(dates[-1].date()),"target":str((dates[-1]+pd.DateOffset(months=h)).date()),
                "predicted":float(pred[i]),"model":"pooled_ridge" if h!=12 else "damped_trend",
                "status":"historical_export_not_current_2026_forecast"})
    pd.DataFrame(records).to_csv(out/"forecast_export.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    ds=json.loads((out/"detector_selection.json").read_text())
    events=pd.read_csv(ROOT/"data/news_events.csv")
    alarms=[];latest=[]
    for num,((tid,cat),row) in enumerate(panel.iterrows(),1):
        scores=detect(row.to_numpy(),ds["method"],ds["thresholds"][ds["method"]],cfg["alarm_cooldown"])
        for r in scores:
            if not (r["alarm"] or r["warning"]):continue
            as_of=dates[r["t"]]+pd.offsets.MonthBegin()
            known=events[(pd.to_datetime(events.available_at)<as_of)&
                         (pd.to_datetime(events.available_at)>=as_of-pd.Timedelta(days=90))&events.delta_pp.ne(0)]
            alarms.append({"territory_id":int(tid),"category":cat,"observation_month":str(dates[r["t"]].date()),
                "as_of":str(as_of.date()),"method":ds["method"],"score":r["score"],"alarm":r["alarm"],
                "warning":r["warning"],"direction":r["direction"],"known_news_ids":";".join(known.event_id),
                "causality":"not_established"})
        r=scores[-1]
        latest.append({"territory_id":int(tid),"category":cat,**{k:r[k] for k in ["score","alarm","warning","direction"]}})
        if num%3000==0:print(f"real detectors: {num}/{len(panel)}",flush=True)
    pd.DataFrame(alarms).to_csv(out/"signals.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    pd.DataFrame(latest).to_csv(out/"latest_signals.csv",index=False)
    print(f"export complete: {len(records)} forecasts and {len(alarms)} signal rows",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["all","confirmation","export","delay"],default="all")
    a=p.parse_args();cfg=json.loads((ROOT/"final_config.json").read_text());rcfg=json.loads((ROOT/"release_config.json").read_text())
    out=ROOT/"final_results"
    if a.phase in ("all","confirmation"):confirmation(cfg,rcfg,out)
    if a.phase in ("all","delay"):delay(cfg,out)
    if a.phase in ("all","export"):export(cfg,out)
