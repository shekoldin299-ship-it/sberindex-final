"""Extra territory check with policy frozen before this experiment."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from detectors import ROOT, load_data, generate, score, detector_cases, detector_score


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    _,p,_,_=load_data(ROOT/'pipeline/data/consumption.parquet')
    p=p.xs('Все категории',level=1).dropna();a=p.to_numpy();z=np.log1p(a)
    factor=np.median(z,axis=0);growth=z[:,12:]-z[:,:-12];gf=np.median(growth,axis=0)
    spread=np.maximum(1.4826*np.median(abs(growth-gf),axis=0),.02)
    order=sorted(range(len(p)),key=lambda i:hashlib.sha256(f'20261011:{p.index[i]}'.encode()).hexdigest())
    ids=order[600:900];seed=20261019
    protocol={'seed':seed,'territory_ids':[int(p.index[i]) for i in ids],
        'method':'confirmed2','threshold':4,'calibration_disjoint':True,'previous_test_disjoint':True}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cases=generate(a,ids,seed,factor);r,d=score(cases,'confirmed2',4);rows=[r]
    old=detector_cases(a,ids,gf,spread,seed)
    for method,t in [('cusum_v1',8),('multiscale',5)]:
        v,_=detector_score(old,method,t);rows.append(v)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'cases.json').write_text(json.dumps(d))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main(sys.argv[1])
