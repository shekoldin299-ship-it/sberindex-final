"""Same V5 annual formula, now explicitly supporting short municipal prefixes."""
from pathlib import Path
import numpy as np,pandas as pd,sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'pipeline'));from benchmark import load_data

def annual_forecast(history,categories,national,origin,lag=1,horizon=12):
 origin=pd.Period(origin,'M');y=np.asarray(history,float)
 if y.ndim!=2 or y.shape[1]<3 or not np.isfinite(y).any(axis=1).all():raise ValueError('Need >=3 calendar months and observed data')
 if lag<1 or not 1<=horizon<=12:raise ValueError('Invalid lag or horizon')
 y=np.maximum(pd.DataFrame(y.T).ffill().bfill().to_numpy().T,1.);n=national.loc[:origin-lag];cats=np.asarray(categories);out=np.zeros(len(y));growth_base=np.zeros(len(y));months=pd.period_range(end=origin,periods=y.shape[1],freq='M');target=origin+horizon
 for cat in sorted(set(cats)):
  mask=cats==cat;field={'Продовольствие':'food','Общественное питание':'catering'}.get(cat,'total');ns=n[field]
  growth=float(np.clip(np.median(np.log(ns/ns.shift(12)).dropna().iloc[-3:]),-.1,.5))
  if not np.isfinite(growth):raise ValueError('National history too short')
  anchor=target-12
  for _ in range(10):
   if anchor in ns.index:break
   anchor-=12
  else:raise ValueError('No national seasonal anchor')
  nt=float(ns.loc[anchor]*np.exp(growth*(target.ordinal-anchor.ordinal)/12))
  nh=np.array([ns.loc[m] if m in ns.index else ns.loc[m-12]*np.exp(growth) for m in months])
  out[mask]=np.median((y[mask]/nh)[:,-3:],axis=1)*nt
  growth_base[mask]=y[mask,-1]*np.exp(growth*horizon/12)
 return out,growth_base

def main():
 out=HERE/'annual_results';out.mkdir(exist_ok=False)
 _,panel,dates,_=load_data(ROOT/'pipeline/data/consumption.parquet');n=pd.read_csv(ROOT/'research/v5/national_source.csv');n.index=pd.PeriodIndex(n.date,freq='M');n['total']=n[['food','nonfood','catering','services']].sum(axis=1)
 rows=[]
 for c in range(3,13):
  s=panel.loc[panel.iloc[:,:c].notna().any(axis=1)];cats=s.index.get_level_values(1);y=s.iloc[:,:c].to_numpy();origin=pd.Period(dates[c-1],'M');pred,gb=annual_forecast(y,cats,n,origin);lag2,_=annual_forecast(y,cats,n,origin,2)
  modified=n.copy();modified.loc[origin:]=1e12;np.testing.assert_array_equal(pred,annual_forecast(y,cats,modified,origin)[0])
  last=pd.DataFrame(y.T).ffill().bfill().iloc[-1].to_numpy()
  d=pd.DataFrame(dict(territory_id=s.index.get_level_values(0),category=cats,origin=c,horizon=12,actual=s.iloc[:,c+11],annual_v5=pred,annual_lag2=lag2,last_value=last,national_growth=gb)).reset_index(drop=True);rows.append(d[np.isfinite(d.actual)])
 d=pd.concat(rows,ignore_index=True);assert not d.duplicated(['territory_id','category','origin','horizon']).any()
 old=pd.read_parquet(ROOT/'research/v5/results/predictions.parquet');old=old[old.horizon==12]
 j=old.merge(d,on=['territory_id','category','origin','horizon'],validate='one_to_one',suffixes=('_old',''))
 assert len(j)==len(old);np.testing.assert_array_equal(j.v5,j.annual_v5);np.testing.assert_array_equal(j.actual_old,j.actual)
 parts=out/'predictions.parquet';parts.mkdir()
 for i,start in enumerate(range(0,len(d),18000)):d.iloc[start:start+18000].to_parquet(parts/f'part-{i:03d}.parquet',index=False)
 metrics=[]
 for scope,x in [('all',d),('total',d[d.category=='Все категории'])]:
  for c,g in [(-1,x)]+list(x.groupby('origin')):
   for m in ['annual_v5','annual_lag2','last_value','national_growth']:
    e=g[m]-g.actual;metrics.append(dict(scope=scope,origin=int(c),n=len(g),model=m,mae=abs(e).mean(),r2=1-(e**2).sum()/((g.actual-g.actual.mean())**2).sum()))
 m=pd.DataFrame(metrics);m.to_csv(out/'metrics.csv',index=False)
 (out/'verification.json').write_text(json.dumps({'status':'PASS','annual_origins':10,'forecast_pairs':len(d),'fixed_horizon_months':12,'min_history':3,'max_history':12,'future_invariance':True,'old_v5_exact_match':True,'new_blind_test':False,'yearly_coverage_limit':'Same 2024 target year; shorter municipal prefixes, current-vintage national history'},indent=2))
 print(m[m.origin==-1].to_string(index=False));print(m[(m.scope=='all')].pivot(index='origin',columns='model',values='mae').round(2).to_string())
if __name__=='__main__':main()
