"""Frozen V2 experiments. Existing outputs remain unchanged; no test-driven switch."""
import argparse,hashlib,json,math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import baseline_forecasts
from forecasting import PooledRidge,fill_history,ranked_ids
from experiment import ROOT,prepare,foundation_pipeline,foundation_predict,metric_table,write_json
from release_candidate import prophet_only
from detection import innovations,synthetic_cases,control_chart

OUT=ROOT/'upgrade_results'
KEY=['territory_id','category','origin','horizon']

def adapted_foundation(pipe,history,h,cfg):
    a=fill_history(history);logs=np.log1p(a[:,:12]);t=np.arange(12);xc=t-t.mean()
    slope=logs@xc/(xc@xc)
    profile=logs-slope[:,None]*t;profile-=profile.mean(axis=1)[:,None]
    season=np.exp(profile)
    adjusted=(a+1)/season[:,np.arange(a.shape[1])%12]
    pred=foundation_predict(pipe,adjusted,h,cfg['foundation_batch_size'])
    restored=pred*season[:,(np.arange(h)+a.shape[1])%12]-1
    return np.maximum(np.clip(restored,.25*a[:,-1,None],4*a[:,-1,None]),0)

def candidates(wide):
    w=wide.copy()
    for weight in (0,.25,.5,.75,1):w[f'blend_{weight:g}']=weight*w.pooled_ridge+(1-weight)*w.seasonal_scaled
    w['chronos_seasonal_mix']=.25*w.chronos_adapted+.75*w.seasonal_scaled
    return w

def choose_policy(wide):
    policy={}
    names=[f'blend_{x:g}' for x in (0,.25,.5,.75,1)]+['chronos_adapted','chronos_seasonal_mix']
    for (h,cat),g in wide[wide.horizon.ne(12)].groupby(['horizon','category']):
        errors={n:float(abs(g[n]-g.actual).mean()) for n in names}
        best=min(names,key=lambda n:errors[n]);base=errors['blend_0']
        if errors[best]>base*.98:best='blend_0'
        policy[f'{h}|{cat}']={'model':best,'tuning_mae':errors[best],'seasonal_mae':base}
    return policy

def apply_policy(w,policy):
    return np.array([r['damped_trend'] if r['horizon']==12 else r[policy[f"{int(r['horizon'])}|{r['category']}"]['model']] for _,r in w.iterrows()])

def wide_from_existing(path,split=None):
    p=pd.read_parquet(path)
    if split:p=p[p.split.isin(split)]
    if 'release_candidate' in set(p.model):
        p=p.copy();p.loc[p.model.eq('release_candidate')&p.horizon.ne(12),'model']='pooled_ridge'
        p.loc[p.model.eq('release_candidate')&p.horizon.eq(12),'model']='damped_trend'
    return p.pivot(index=KEY+['actual','history_scale'],columns='model',values='predicted').reset_index()

def add_adapted(w,panel,pipe,cfg):
    w=w.copy();w['chronos_adapted']=np.nan
    for c,g in w.groupby('origin'):
        idx=pd.MultiIndex.from_frame(g[['territory_id','category']].drop_duplicates())
        history=panel.loc[idx].iloc[:,:c].to_numpy();pred=adapted_foundation(pipe,history,int(g.horizon.max()),cfg)
        lookup={key:i for i,key in enumerate(idx)}
        for row in g.itertuples():w.loc[row.Index,'chronos_adapted']=pred[lookup[(row.territory_id,row.category)],row.horizon-1]
    # h12 only uses damped_trend; fill irrelevant missing ridge to keep mixtures finite.
    if 'pooled_ridge' not in w:w['pooled_ridge']=w.damped_trend
    w['pooled_ridge']=w.pooled_ridge.fillna(w.get('damped_trend',w.seasonal_scaled))
    return candidates(w)

def quantile(scores,coverage=.9):
    a=np.sort(np.asarray(scores));k=math.ceil((len(a)+1)*coverage)
    if k>len(a):raise ValueError('Not enough calibration territories')
    return float(a[k-1])

def forecasting():
    cfg=json.loads((ROOT/'final_config.json').read_text())
    _,panel,dates,cats,_,training,old_ids,_=prepare(cfg)
    pilot=json.loads((ROOT/'results/results.json').read_text())['pilot_territory_ids']
    calids=json.loads((ROOT/'final_results/confirmation_protocol.json').read_text())['ids']
    ids=ranked_ids(panel,20261010,pilot+old_ids+calids)[:100]
    signature={'code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol':hashlib.sha256((ROOT/'UPGRADE_PROTOCOL.md').read_bytes()).hexdigest(),'test_ids':ids,'tuning_ids':old_ids,'calibration_ids':calids}
    write_json(OUT/'protocol.json',signature)
    pipe=foundation_pipeline(cfg)
    tune=add_adapted(wide_from_existing(ROOT/'final_results/paired_predictions.parquet',['test']),panel,pipe,cfg)
    policy=choose_policy(tune);write_json(OUT/'policy.json',policy)
    tune.to_parquet(OUT/'tuning_predictions.parquet',index=False)
    print('Policy frozen',json.dumps(policy,ensure_ascii=False),flush=True)
    cal=add_adapted(wide_from_existing(ROOT/'final_results/confirmation_predictions.parquet'),panel,pipe,cfg)
    cal['v2']=apply_policy(cal,policy);cal['score']=abs(cal.v2-cal.actual)/cal.history_scale
    widths={}
    for (h,cat),g in cal.groupby(['horizon','category']):
        s=g.groupby('territory_id').score.max();widths[f'{h}|{cat}']={'q':quantile(s),'territories':len(s)}
    write_json(OUT/'interval_calibration.json',widths);cal.to_parquet(OUT/'calibration_predictions.parquet',index=False)
    sample=panel.loc[panel.index.get_level_values(0).isin(ids)];codes=np.array([cats.index(c) for _,c in sample.index]);tcodes=np.array([cats.index(c) for _,c in training.index]);rows=[]
    for c in (12,18,21,23):
        hs=[12] if c==12 else [h for h in (1,3,6) if c+h<=24]
        history=sample.iloc[:,:c].to_numpy();adapt=adapted_foundation(pipe,history,max(hs),cfg);raw=foundation_predict(pipe,history,max(hs),64)
        ridge=PooledRidge(cfg,cats).fit(training.iloc[:,:c].to_numpy(),tcodes,dates)
        print(f'V2 new cohort origin {c}: fitting Prophet for {len(history)} series',flush=True)
        with ProcessPoolExecutor(max_workers=4) as ex:prophets=np.array(list(ex.map(prophet_only,[(dates[:c],y,hs,cfg['seed']) for y in history])))
        for j,h in enumerate(hs):
            rp=ridge.predict(history,codes,dates,h)
            for i,(tid,cat) in enumerate(sample.index):
                actual=sample.iloc[i,c+h-1]
                if not np.isfinite(actual):continue
                base=baseline_forecasts(history[i],h)
                rows.append({'territory_id':int(tid),'category':cat,'origin':c,'horizon':h,'actual':float(actual),'history_scale':float(max(np.nanmean(history[i]),1)),
                 'pooled_ridge':rp[i],'seasonal_scaled':base['seasonal_scaled'],'damped_trend':base['damped_trend'],'chronos_adapted':adapt[i,h-1],'chronos_bolt_tiny':raw[i,h-1],'prophet_default':prophets[i,j]})
        print(f'V2 origin {c} complete',flush=True)
    w=candidates(pd.DataFrame(rows));w['v2']=apply_policy(w,policy);w['v1']=np.where(w.horizon.eq(12),w.damped_trend,w.pooled_ridge)
    q=np.array([widths[f'{r.horizon}|{r.category}']['q'] for r in w.itertuples()]);w['lower']=np.maximum(w.v2-q*w.history_scale,0);w['upper']=w.v2+q*w.history_scale
    w.to_parquet(OUT/'test_predictions.parquet',index=False)
    names=['v2','v1','seasonal_scaled','prophet_default','chronos_bolt_tiny','chronos_adapted']
    long=w.melt(id_vars=KEY+['actual','history_scale'],value_vars=names,var_name='model',value_name='predicted')
    metric_table(long,['model','horizon']).to_csv(OUT/'metrics.csv',index=False);metric_table(long,['model','horizon','category']).to_csv(OUT/'category_metrics.csv',index=False)
    interval=[]
    for h,g in w.groupby('horizon'):
        covered=g.actual.between(g.lower,g.upper)
        joint=g.assign(covered=covered).groupby(['territory_id','category']).covered.all()
        interval.append({'horizon':int(h),'n':len(g),'coverage':float(covered.mean()),'joint_territory_category_coverage':float(joint.mean()),'mean_width':float((g.upper-g.lower).mean())})
    write_json(OUT/'interval_metrics.json',interval)
    history=panel.to_numpy();codes=np.array([cats.index(c) for _,c in panel.index]);ridge=PooledRidge(cfg,cats).fit(training.to_numpy(),tcodes,dates)
    needs=any('chronos' in x['model'] for x in policy.values());adapt=adapted_foundation(pipe,history,6,cfg) if needs else None
    exports=[]
    for h in (1,3,6,12):
        rp=ridge.predict(history,codes,dates,h)
        for i,(tid,cat) in enumerate(panel.index):
            base=baseline_forecasts(history[i],h);name='damped_trend' if h==12 else policy[f'{h}|{cat}']['model']
            if name.startswith('blend_'):
                weight=float(name.split('_')[1]);pred=weight*rp[i]+(1-weight)*base['seasonal_scaled']
            elif name=='chronos_adapted':pred=adapt[i,h-1]
            elif name=='chronos_seasonal_mix':pred=.25*adapt[i,h-1]+.75*base['seasonal_scaled']
            else:pred=base['damped_trend']
            width=widths[f'{h}|{cat}']['q']*max(np.nanmean(history[i]),1)
            exports.append({'territory_id':int(tid),'category':cat,'horizon':h,'target':str((dates[-1]+pd.DateOffset(months=h)).date()),'predicted':float(pred),'lower':float(max(0,pred-width)),'upper':float(pred+width),'model':name})
    pd.DataFrame(exports).to_csv(OUT/'forecast_export.csv.gz',index=False,compression={'method':'gzip','mtime':0});print('V2 forecasts complete',flush=True)

def robust_chart(z,method,threshold):
    if method=='cusum_v1':return control_chart(z,'cusum',8,3)
    window=[];last=-100;out=[]
    for t,value in enumerate(z):
        if not np.isfinite(value):out.append({'t':t,'score':0.,'warning':False,'alarm':False,'direction':0});continue
        window.append(float(np.clip(value,-3,3)))
        if method=='median3':stat=float(np.median(window[-3:])*np.sqrt(3)) if len(window)>=3 else 0.
        elif method=='multiscale':
            stats=[sum(window[-k:])/np.sqrt(k) for k in range(2,min(6,len(window))+1)]
            stat=max(stats,key=abs) if stats else 0.
        else:raise ValueError(method)
        score=abs(stat)/threshold;alarm=bool(score>=1 and t-last>6)
        out.append({'t':t,'score':score,'warning':bool(score>=.7),'alarm':alarm,'direction':int(np.sign(stat))})
        if alarm:last=t;window=[]
    return out

def score_cases(cases,method,threshold):
    tp=fp=fn=nulls=0;delays=[];details=[]
    for case in cases:
        alarms=[r['t'] for r in robust_chart(case['z'],method,threshold) if r['alarm']];change=case['change']
        hits=[] if change is None else [t for t in alarms if change<=t<=change+6];hit=bool(hits)
        tp+=hit;fp+=len(alarms)-hit;fn+=change is not None and not hit
        if hit:delays.append(hits[0]-change)
        nulls+=case['kind']=='none' and bool(alarms)
        details.append({'id':case['id'],'kind':case['kind'],'change':change,'alarms':alarms})
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return {'method':method,'threshold':threshold,'tp':int(tp),'fp':int(fp),'fn':int(fn),'precision':p,'recall':r,'f1':2*p*r/max(p+r,1e-12),'null_rate':nulls/100,'median_delay':float(np.median(delays)) if delays else None},details

def detectors():
    def cases(seed,length):
        a=synthetic_cases(seed,100,length)
        for c in a:c['z']=innovations(c['y'])
        return a
    calibration={length:cases(2042,length) for length in (24,60)};rows=[]
    for method in ('cusum_v1','median3','multiscale'):
        for threshold in ([8] if method=='cusum_v1' else [2,3,4,5,6,8]):
            for length,c in calibration.items():
                r,_=score_cases(c,method,threshold);rows.append({**r,'length':length,'split':'calibration'})
    scores=pd.DataFrame(rows);summary=scores.groupby(['method','threshold']).agg(f1=('f1','mean'),max_null=('null_rate','max')).reset_index()
    eligible=summary[summary.max_null.le(.1)]
    best=(eligible.sort_values(['f1','method'],ascending=[False,True]) if len(eligible) else summary.sort_values(['max_null','f1'],ascending=[True,False])).iloc[0]
    selected={'method':best['method'],'threshold':float(best.threshold),'calibration_mean_f1':float(best.f1)};write_json(OUT/'detector_selection.json',selected)
    details=[]
    for length in (24,60):
        c=cases(3042,length)
        for method,threshold in {('cusum_v1',8),(selected['method'],selected['threshold'])}:
            r,detail=score_cases(c,method,threshold);rows.append({**r,'length':length,'split':'test'})
            details.append({'method':method,'length':length,'cases':detail})
    pd.DataFrame(rows).to_csv(OUT/'detector_metrics.csv',index=False);write_json(OUT/'detector_cases.json',details)
    print('V2 detectors complete',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['forecast','detector'],required=True);a=p.parse_args();OUT.mkdir(exist_ok=True)
    forecasting() if a.phase=='forecast' else detectors()
