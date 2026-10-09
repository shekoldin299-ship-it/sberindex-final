"""Independent integrity checks for frozen V2 selection, targets and intervals."""
import hashlib,json
import numpy as np
import pandas as pd
from benchmark import load_data
from experiment import ROOT,write_json
from upgrade_v2 import OUT,choose_policy,apply_policy,quantile

def validate():
    sig=json.loads((OUT/'protocol.json').read_text())
    assert hashlib.sha256((ROOT/'upgrade_v2.py').read_bytes()).hexdigest()==sig['code']
    assert hashlib.sha256((ROOT/'UPGRADE_PROTOCOL.md').read_bytes()).hexdigest()==sig['protocol']
    a,b,c=map(set,[sig['tuning_ids'],sig['calibration_ids'],sig['test_ids']]);assert not (a&b or a&c or b&c)
    t=pd.read_parquet(OUT/'tuning_predictions.parquet');policy=json.loads((OUT/'policy.json').read_text());assert choose_policy(t)==policy
    p=pd.read_parquet(OUT/'test_predictions.parquet');assert not p.duplicated(['territory_id','category','origin','horizon']).any()
    np.testing.assert_allclose(p.v2,apply_policy(p,policy),rtol=1e-12)
    _,panel,dates,_=load_data(ROOT/'data/consumption.parquet')
    for r in p.itertuples():assert np.isclose(r.actual,panel.loc[(r.territory_id,r.category)].iloc[r.origin+r.horizon-1],rtol=1e-12)
    widths=json.loads((OUT/'interval_calibration.json').read_text());cal=pd.read_parquet(OUT/'calibration_predictions.parquet')
    for (h,cat),g in cal.groupby(['horizon','category']):
        scores=(abs(g.v2-g.actual)/g.history_scale).groupby(g.territory_id).max()
        assert np.isclose(quantile(scores),widths[f'{h}|{cat}']['q'])
    for r in p.itertuples():
        q=widths[f'{r.horizon}|{r.category}']['q']*r.history_scale
        assert np.isclose(r.lower,max(0,r.v2-q)) and np.isclose(r.upper,r.v2+q)
    for r in pd.read_csv(OUT/'metrics.csv').itertuples():
        g=p[p.horizon.eq(r.horizon)];assert np.isclose(abs(g.actual-g[r.model]).mean(),r.mae,rtol=1e-10)
    dm=pd.read_csv(OUT/'detector_metrics.csv')
    for group in json.loads((OUT/'detector_cases.json').read_text()):
        tp=fp=fn=0
        for case in group['cases']:
            hits=[] if case['change'] is None else [t for t in case['alarms'] if case['change']<=t<=case['change']+6]
            hit=bool(hits);tp+=hit;fp+=len(case['alarms'])-hit;fn+=case['change'] is not None and not hit
        row=dm[dm.split.eq('test')&dm.method.eq(group['method'])&dm.length.eq(group['length'])].iloc[0]
        assert (tp,fp,fn)==(row.tp,row.fp,row.fn)
    f=pd.read_csv(OUT/'forecast_export.csv.gz');assert len(f)==len(panel)*4
    assert np.isfinite(f[['predicted','lower','upper']]).all().all() and (f.lower<=f.predicted).all() and (f.upper>=f.predicted).all()
    report={'status':'PASS','test_cases':len(p),'test_territories_with_targets':int(p.territory_id.nunique()),'export_rows':len(f),'checks':['frozen source hash','disjoint geographic cohorts','tuning-only selection','source targets','point forecasts','calibration-only interval quantiles','recomputed MAE','detector event scoring','nonnegative finite exports']}
    write_json(OUT/'integrity.json',report);print(report)

if __name__=='__main__':validate()
