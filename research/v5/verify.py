"""Verify frozen V5 evidence without network or fitting."""
import json,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
    out=HERE/'results';d=pd.read_parquet(out/'predictions.parquet')
    assert not d.duplicated(['territory_id','category','origin','horizon']).any()
    assert np.isfinite(d[['actual','v4','v5','v5_lag2']]).all().all()
    for p,h in json.loads((out/'signature.json').read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    m=pd.read_csv(out/'metrics.csv')
    for _,r in m.iterrows():
        g=d[d.horizon==r.horizon]
        if r.scope=='total_spending':g=g[g.category=='Все категории']
        assert len(g)==r.n
        np.testing.assert_allclose(abs(g[r.model]-g.actual).mean(),r.mae,rtol=1e-12)
    np.testing.assert_array_equal(d[d.horizon.isin([3,6])].v4,d[d.horizon.isin([3,6])].v5)
    source=pd.read_parquet(ROOT/'pipeline/data/consumption.parquet')
    source['month']=pd.PeriodIndex(source.date,freq='M').astype(str)
    d['month']=[str(pd.Period('2023-01','M')+int(c+h-1)) for c,h in zip(d.origin,d.horizon)]
    value='value' if 'value' in source else 'consumption'
    j=d.merge(source[['territory_id','category','month',value]],on=['territory_id','category','month'],validate='many_to_one')
    assert len(j)==len(d);np.testing.assert_array_equal(j.actual,j[value])
    print(f'PASS V5: {len(d)} forecast pairs, input signatures, metrics and actuals; no training.')
if __name__=='__main__':main()
