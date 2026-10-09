from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
 for p,h in json.loads((HERE/'signature.json').read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 d=pd.read_parquet(HERE/'annual_results/predictions.parquet');K=['territory_id','category','origin','horizon']
 assert not d.duplicated(K).any();assert d.horizon.eq(12).all();assert sorted(d.origin.unique())==list(range(3,13));assert len(d)==124284
 cols=['actual','annual_v5','annual_lag2','last_value','national_growth'];assert np.isfinite(d[cols]).all().all()
 raw=pd.read_parquet(ROOT/'pipeline/data/consumption.parquet');raw['month']=pd.PeriodIndex(raw.date,freq='M').astype(str);d['month']=[str(pd.Period('2023-01','M')+int(c)+11) for c in d.origin]
 value='value' if 'value' in raw else 'consumption';j=d.merge(raw[['territory_id','category','month',value]],on=['territory_id','category','month'],validate='many_to_one');assert len(j)==len(d);np.testing.assert_array_equal(j.actual,j[value])
 m=pd.read_csv(HERE/'annual_results/metrics.csv')
 for _,r in m.iterrows():
  g=d if r.scope=='all' else d[d.category=='Все категории']
  if r.origin!=-1:g=g[g.origin==r.origin]
  assert len(g)==r.n;np.testing.assert_allclose(abs(g[r.model]-g.actual).mean(),r.mae,rtol=1e-12)
 old=pd.read_parquet(ROOT/'research/v5/results/predictions.parquet');old=old[old.horizon==12];j=d.merge(old[K+['v5']],on=K,validate='one_to_one');assert len(j)==len(old);np.testing.assert_array_equal(j.annual_v5,j.v5)
 print('PASS 124284 annual pairs, 10 fixed-h12 origins, all facts, MAE, hashes, exact V5 match')
if __name__=='__main__':main()
