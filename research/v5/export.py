"""Historical V5 forecast from December 2024. Not a current 2026 forecast."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from run import ROOT,HERE,load_national
from model import forecast,load_config
from benchmark import load_data

def main(output,config_path=None):
    config=load_config(config_path)
    output=Path(output)
    if output.exists():raise ValueError('Output must not exist')
    _,panel,dates,_=load_data(ROOT/'pipeline/data/consumption.parquet')
    source=ROOT/'research/v4/models.py';spec=importlib.util.spec_from_file_location('v4_model',source);v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
    categories=np.array(panel.index.get_level_values(1));codes=pd.Categorical(categories,categories=sorted(set(categories))).codes
    model=v4.PanelModel(panel.to_numpy(),codes);pred=model.predict(1)
    previous=[]
    for p in sorted((ROOT/'research/v4/results/forecast').glob('origin_*.parquet')):
        f=pd.read_parquet(p);f=f[f.horizon==1]
        for cat,g in f.groupby('category'):
            previous.append(dict(category=cat,horizon=1,target_index=int(g.origin.iloc[0]),**{n:float(abs(g[n]-g.actual).mean()) for n in v4.MEMBERS}))
    past=pd.concat([pd.read_parquet(ROOT/f'research/v4/results/forecast/origin_{c}.parquet').query('horizon==1') for c in [21,22,23]])
    old=pd.concat([pd.read_parquet(ROOT/f'research/v4/results/expanded/origin_{c}.parquet') for c in [21,22,23]])
    past=past.merge(old[['territory_id','category','origin','horizon','v3']],on=['territory_id','category','origin','horizon'],validate='one_to_one')
    weights={};adopt={}
    for cat,g in past.groupby('category'):
        a=g.groupby('origin').apply(lambda q:pd.Series({'v3':abs(q.v3-q.actual).mean(),'mix':abs(q.online_mix-q.actual).mean()}),include_groups=False)
        adopt[cat]=a['mix'].mean()<=.98*a.v3.mean()
        weights[cat]=v4.mixture_weights(previous,cat,1,24)
    oldexport=pd.read_csv(ROOT/'pipeline/v3_results/forecast_export.csv.gz');frames=[]
    for h,g in oldexport.groupby('horizon'):
        idx=panel.index.get_indexer(pd.MultiIndex.from_frame(g[['territory_id','category']]))
        assert (idx>=0).all()
        base=g.predicted.to_numpy().copy()
        if h==1:
            for cat in sorted(set(categories)):
                mask=g.category.to_numpy()==cat
                if adopt[cat]:
                    mix=np.column_stack([pred[n][idx[mask]] for n in v4.MEMBERS])@weights[cat]
                    base[mask]=.5*base[mask]+.5*mix
        fullbase=np.zeros(len(panel));fullbase[idx]=base
        new=forecast(panel.to_numpy(),categories,load_national(),'2024-12',int(h),fullbase,config=config)
        out=g[['territory_id','category','horizon','target']].copy();out['predicted']=new[idx];out['model']='V5';out['origin']='2024-12-01'
        frames.append(out)
    final=pd.concat(frames,ignore_index=True)
    assert len(final)==52560 and np.isfinite(final.predicted).all()
    output.parent.mkdir(parents=True,exist_ok=True);final.to_csv(output,index=False,compression='gzip')
    print('PASS V5 export',len(final),'historical forecasts; intervals not recalibrated/not included')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config');args=p.parse_args();main(args.output,args.config)
