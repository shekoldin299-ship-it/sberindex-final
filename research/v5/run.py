"""Recompute V5 correction from frozen V4 forecasts and source data, CPU/offline."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from model import forecast,load_config
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from benchmark import load_data

def load_national():
    n=pd.read_csv(HERE/'national_source.csv');n.index=pd.PeriodIndex(n.date,freq='M')
    n['total']=n[['food','nonfood','catering','services']].sum(axis=1)
    return n

def baseline():
    frames=[pd.read_parquet(p) for p in sorted((ROOT/'research/v4/results/safe').glob('origin_*.parquet'))]
    one=pd.concat(frames,ignore_index=True).rename(columns={'safe_mix':'v4'})
    old=pd.read_parquet(ROOT/'pipeline/v3_results/predictions.parquet')
    old=old[old.horizon!=1].rename(columns={'v3':'v4'})
    cols=['territory_id','category','origin','horizon','actual','v4']
    return pd.concat([one[cols],old[cols]],ignore_index=True)

def main(out,config_path=None):
    config_path=Path(config_path or HERE/'config.json').resolve()
    config=load_config(config_path)
    out=Path(out).resolve()
    if out.exists():raise ValueError('Output must be a new directory')
    out.mkdir(parents=True)
    (out/'effective_config.json').write_text(json.dumps(config,indent=2)+'\n')
    _,panel,dates,_=load_data(ROOT/'pipeline/data/consumption.parquet');national=load_national()
    b=baseline();frames=[]
    for (c,h),g in b.groupby(['origin','horizon']):
        # Compute panel factors over all observable rows, not the evaluation subset.
        sample=panel.loc[panel.iloc[:,:c].notna().any(axis=1)]
        lookup=pd.MultiIndex.from_frame(g[['territory_id','category']]);idx=sample.index.get_indexer(lookup)
        assert (idx>=0).all()
        fullbase=np.zeros(len(sample));fullbase[idx]=g.v4.to_numpy()
        g=g.copy()
        for lag,name in [(1,'v5'),(2,'v5_lag2')]:
            pred=forecast(sample.iloc[:,:c].to_numpy(),sample.index.get_level_values(1),national,pd.Period(dates[c-1],freq='M'),int(h),fullbase,lag,config)
            g[name]=pred[idx]
        np.testing.assert_array_equal(g.actual.to_numpy(),sample.iloc[idx,c+h-1].to_numpy())
        frames.append(g)
    full=pd.concat(frames,ignore_index=True)
    assert not full.duplicated(['territory_id','category','origin','horizon']).any()
    assert np.isfinite(full[['actual','v4','v5','v5_lag2']]).all().all()
    parts=out/'predictions.parquet';parts.mkdir()
    for i,start in enumerate(range(0,len(full),18000)):full.iloc[start:start+18000].to_parquet(parts/f'part-{i:03d}.parquet',index=False)
    rows=[]
    for scope,d in [('six_categories',full),('total_spending',full[full.category=='Все категории'])]:
        for h,g in d.groupby('horizon'):
            for name in ['v4','v5','v5_lag2']:
                e=g[name]-g.actual;sst=((g.actual-g.actual.mean())**2).sum()
                rows.append(dict(scope=scope,horizon=int(h),model=name,n=len(g),origins=g.origin.nunique(),mae=abs(e).mean(),r2=1-(e**2).sum()/sst,wape=abs(e).sum()/g.actual.abs().sum()))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    monthly=[]
    for (c,h,cat),g in full.groupby(['origin','horizon','category']):
        monthly.append(dict(origin=int(c),horizon=int(h),category=cat,n=len(g),**{m:abs(g[m]-g.actual).mean() for m in ['v4','v5','v5_lag2']}))
    pd.DataFrame(monthly).to_csv(out/'by_month_category.csv',index=False)
    sig={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'model.py',HERE/'run.py',HERE/'national_source.csv',ROOT/'pipeline/data/consumption.parquet']}
    (out/'signature.json').write_text(json.dumps(sig,indent=2))
    (out/'completion.json').write_text(json.dumps({'status':'retrospective_design_after_comparison_not_blind_test','baseline_commit':'4691e2606c13e3891a9d9bd3249c065b16c44567','pairs':len(full),'national_lag_months':1,'sensitivity_lag_months':2},indent=2))
    print(metrics.pivot(index=['scope','model'],columns='horizon',values='mae').round(2).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config');args=p.parse_args();main(args.output,args.config)
