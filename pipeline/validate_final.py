"""Independent output integrity checks; no fit-time inference from green checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import load_data

ROOT=Path(__file__).parent


def validate(out,require_export=True):
    config=json.loads((ROOT/"final_config.json").read_text())
    signature=json.loads((out/"run_signature.json").read_text())
    assert signature["data"]==hashlib.sha256((ROOT/"data/consumption.parquet").read_bytes()).hexdigest()
    df,panel,dates,_=load_data(ROOT/"data/consumption.parquet")
    cohort=json.loads((out/"cohort.json").read_text())
    p=pd.read_parquet(out/"paired_predictions.parquet")
    keys=["territory_id","category","origin","horizon","target"]
    assert not p.duplicated(["model"]+keys).any(),"Duplicate forecast cases"
    models=sorted(p.model.unique())
    assert len(models)==9,"Missing forecast model"
    counts=p.groupby(keys,dropna=False).model.nunique()
    assert counts.eq(len(models)).all(),"Unpaired forecasts"
    assert np.isfinite(p[["predicted","actual","history_scale"]]).all().all()
    assert p.predicted.ge(0).all() and p.history_scale.gt(0).all()
    truths={(int(r.territory_id),r.category,r.date):float(r.y) for r in df.itertuples()}
    for r in p.itertuples():
        assert r.actual==truths[r.territory_id,r.category,r.target[:7]]
        assert pd.Timestamp(r.target)==dates[r.origin+r.horizon-1]
        assert pd.Timestamp(r.as_of)==dates[r.origin]
        if r.split=="validation":assert r.origin+r.horizon-1<=17 and r.horizon!=12
        if r.split=="test":assert r.origin>=18 and r.horizon!=12
    for territory in cohort["evaluation_ids"]:
        assert panel.loc[territory].iloc[:,:12].notna().sum(axis=1).ge(9).all()
    metrics=pd.read_csv(out/"forecast_metrics.csv")
    for r in metrics.itertuples():
        g=p[p.split.eq(r.split)&p.model.eq(r.model)&p.horizon.eq(r.horizon)]
        assert len(g)==r.n
        assert np.isclose(np.abs(g.actual-g.predicted).mean(),r.mae,rtol=1e-10)
    choice=json.loads((out/"selection.json").read_text())["by_horizon"]
    for h in (1,3,6):
        val=metrics[metrics.split.eq("validation")&metrics.horizon.eq(h)]
        assert choice[str(h)]==val.sort_values(["mae","model"]).iloc[0].model
    assert choice["12"]==config["h12_deployment_model"]
    det=pd.read_csv(out/"detector_metrics.csv")
    cases=json.loads((out/"synthetic_case_results.json").read_text())
    assert set(det[det.split.eq("test")].length)=={24,60}
    assert set(det.method)=={"cusum","ewma","window"}
    for length in (24,60):
        for method in det.method.unique():
            rows=[r for r in cases if r["length"]==length and r["method"]==method]
            metric=det[det.split.eq("test")&det.length.eq(length)&det.method.eq(method)].iloc[0]
            assert sum(r["detected"] for r in rows)==metric.tp
            for r in rows:
                if r["detected"]:assert 0<=r["delay"]<=config["detection_tolerance"]
    export_rows=0
    confirmation_rows=0
    confirmation=out/"confirmation_predictions.parquet"
    if confirmation.exists():
        cp=pd.read_parquet(confirmation)
        assert not cp.duplicated(["model"]+keys).any()
        assert cp.groupby(keys).model.nunique().eq(4).all()
        assert np.isfinite(cp[["actual","predicted"]]).all().all() and cp.predicted.ge(0).all()
        protocol=json.loads((out/"confirmation_protocol.json").read_text())
        assert not set(protocol["ids"]) & set(cohort["evaluation_ids"])
        pilot=json.loads((ROOT/"results/results.json").read_text())["pilot_territory_ids"]
        assert not set(protocol["ids"]) & set(pilot)
        assert protocol["code"]==hashlib.sha256((ROOT/"release_candidate.py").read_bytes()).hexdigest()
        assert protocol["config"]==hashlib.sha256((ROOT/"release_config.json").read_bytes()).hexdigest()
        for r in cp.itertuples():
            assert r.actual==truths[r.territory_id,r.category,r.target[:7]]
            assert pd.Timestamp(r.target)==dates[r.origin+r.horizon-1]
        cm=pd.read_csv(out/"confirmation_metrics.csv")
        for r in cm.itertuples():
            g=cp[cp.split.eq(r.split)&cp.model.eq(r.model)&cp.horizon.eq(r.horizon)]
            assert len(g)==r.n
            assert np.isclose(abs(g.actual-g.predicted).mean(),r.mae,rtol=1e-10)
        confirmation_rows=len(cp)
    if require_export:
        f=pd.read_csv(out/"forecast_export.csv.gz")
        assert len(f)==len(panel)*4
        assert not f.duplicated(["territory_id","category","horizon"]).any()
        assert np.isfinite(f.predicted).all() and f.predicted.ge(0).all()
        assert set(f.target)=={"2025-01-01","2025-03-01","2025-06-01","2025-12-01"}
        export_rows=len(f)
        signals=pd.read_csv(out/"signals.csv.gz")
        events=pd.read_csv(ROOT/"data/news_events.csv").set_index("event_id")
        for r in signals.itertuples():
            if pd.notna(r.known_news_ids):
                for event in r.known_news_ids.split(";"):
                    assert pd.Timestamp(events.loc[event,"available_at"])<pd.Timestamp(r.as_of)
    report={"status":"PASS","models":models,"prediction_rows":len(p),"unique_paired_cases":len(counts),
            "evaluation_territories":len(cohort["evaluation_ids"]),"export_rows":export_rows,
            "confirmation_prediction_rows":confirmation_rows,
            "checks":["source checksum","paired keys","targets and dates","finite nonnegative forecasts",
                      "recomputed MAE","validation-only selector","independent synthetic event scoring",
                      "news publication-time availability"],
            "limitation":"Integrity checks do not establish statistical superiority or competition acceptance."}
    if require_export:report["checks"].append("export completeness")
    if confirmation_rows:report["checks"].append("confirmation cohort disjointness, frozen code and recomputed MAE")
    (out/"integrity.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"final_results")
    parser.add_argument("--no-export",action="store_true")
    args=parser.parse_args()
    validate(args.output,not args.no_export)
