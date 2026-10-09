"""Verify stored research results independently of presentation tables."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def verify():
    r=HERE/'results';key=['territory_id','category','origin','horizon']
    count=0
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        sha,name=line.split('  ',1);p=(HERE/name).resolve()
        assert p.is_relative_to(HERE) and p.is_file()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
        count+=1
    pred=pd.concat([pd.read_parquet(p) for p in sorted((r/'forecast').glob('origin_*.parquet'))])
    assert len(pred)==375432 and not pred.duplicated(key).any()
    assert (pred.target_index==pred.origin+pred.horizon-1).all()
    table=pd.read_csv(r/'forecast/metrics.csv')
    for row in table.itertuples():
        g=pred[pred.horizon==row.horizon]
        assert len(g)==row.n
        np.testing.assert_allclose(abs(g[row.model]-g.actual).mean(),row.mae,rtol=1e-10)
    safe=pd.concat([pd.read_parquet(p) for p in sorted((r/'safe').glob('origin_*.parquet'))])
    old=pd.concat([pd.read_parquet(p) for p in sorted((r/'expanded').glob('origin_*.parquet'))])
    joined=safe.merge(old,on=key,validate='one_to_one',suffixes=('','_old'))
    np.testing.assert_array_equal(joined.actual,joined.actual_old)
    assert len(joined)==150228
    summary=json.loads((r/'safe/comparison.json').read_text())
    for model in ['v3','safe_mix']:
        np.testing.assert_allclose(abs(joined[model]-joined.actual).mean(),summary[model],rtol=1e-10)
    # Derive detector counts from saved individual alarm timestamps.
    cases=json.loads((r/'confirmation/cases.json').read_text())
    tp=fp=fn=0;controls=0
    for c in cases:
        hits=[] if c['change'] is None else [t for t in c['alarms'] if c['change']<=t<=c['change']+2]
        hit=bool(hits);tp+=hit;fp+=len(c['alarms'])-hit;fn+=c['change'] is not None and not hit
        controls+=c['kind']=='control' and bool(c['alarms'])
    row=pd.read_csv(r/'confirmation/metrics.csv').query('method=="confirmed2"').iloc[0]
    assert (tp,fp,fn)==(row.tp,row.fp,row.fn)
    assert controls==5
    print(f'PASS: {count} hashes, {len(pred)} candidate pairs, {len(joined)} paired updates, {len(cases)} detector cases')


if __name__=='__main__':verify()
