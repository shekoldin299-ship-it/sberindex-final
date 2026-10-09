"""Online control charts. Alarm timestamps are observation times, never backdated."""
import numpy as np


def innovations(values):
    y = np.log1p(np.asarray(values, dtype=float))
    errors, result = [], []
    for t in range(len(y)):
        hist = y[:t]
        valid = np.flatnonzero(np.isfinite(hist))
        z = np.nan
        if t >= 6 and len(valid)>=6 and np.isfinite(y[t]):
            if t>=12 and np.isfinite(y[t-12]):
                ratios = hist[12:]-hist[:-12]
                ratios = ratios[np.isfinite(ratios)]
                growth = np.median(ratios[-6:]) if len(ratios) else 0.0
                pred = y[t-12]+growth
            else:
                recent = valid[-6:]
                slope = np.polyfit(recent,hist[recent],1)[0]
                pred = hist[valid[-1]]+0.8*slope*(t-valid[-1])
            e = y[t]-pred
            if len(errors)>=6:
                ref = np.array(errors[-12:])
                center = np.median(ref)
                scale = max(1.4826*np.median(np.abs(ref-center)),0.03)
                z = np.clip((e-center)/scale,-12,12)
            errors.append(e)
        result.append(float(z))
    return np.array(result)


def control_chart(z, method, threshold, cooldown=3):
    plus = minus = ewma = 0.0
    window, out = [], []
    last_alarm = -100
    for t,value in enumerate(z):
        if not np.isfinite(value):
            out.append({"t":t,"score":0.0,"warning":False,"alarm":False,"direction":0})
            continue
        plus=max(0,plus+value-0.5)
        minus=max(0,minus-value-0.5)
        ewma=0.3*value+0.7*ewma
        window.append(value)
        if method=="cusum":
            score=max(plus,minus)/threshold
            direction=1 if plus>=minus else -1
        elif method=="ewma":
            score=abs(ewma)/(np.sqrt(0.3/1.7)*threshold)
            direction=int(np.sign(ewma))
        elif method=="window":
            score=abs(np.mean(window[-3:]))*np.sqrt(min(len(window),3))/threshold
            direction=int(np.sign(np.mean(window[-3:])))
        else:
            raise ValueError(method)
        alarm=bool(score>=1 and t-last_alarm>cooldown)
        out.append({"t":t,"score":float(score),"warning":bool(score>=0.7),
                    "alarm":alarm,"direction":direction})
        if alarm:
            last_alarm=t
            plus=minus=ewma=0.0
            window=[]
    return out


def detect(values, method, threshold, cooldown=3):
    return control_chart(innovations(values),method,threshold,cooldown)


def synthetic_cases(seed, n, length=60):
    rng=np.random.default_rng(seed)
    cases=[]
    for kind in ["none","outlier","up","down","slope"]:
        for i in range(n):
            t=np.arange(length)
            change=int(rng.integers(14,18) if length==24 else rng.integers(28,43))
            sigma=rng.uniform(0.02,0.07)
            baseline=rng.uniform(6,10)+rng.uniform(-0.002,0.01)*t
            seasonal=rng.uniform(0.03,0.16)*np.sin(2*np.pi*t/12+rng.uniform(0,6.28))
            y=baseline+seasonal+rng.normal(0,sigma,length)
            size=rng.uniform(0.12,0.35)
            if kind=="up": y[change:]+=size
            if kind=="down": y[change:]-=size
            if kind=="slope": y[change:]+=np.arange(length-change)*rng.uniform(0.015,0.035)
            if kind=="outlier": y[change]+=size*2
            cases.append({"id":f"{seed}-{length}-{kind}-{i}","kind":kind,
                          "change":change if kind not in ("none","outlier") else None,
                          "y":np.expm1(y)})
    return cases


def evaluate_detector(cases, method, threshold, cooldown=3, tolerance=6):
    tp=fp=fn=null_alarms=null_n=0
    delays=[]
    detailed=[]
    for case in cases:
        z=case["z"] if "z" in case else innovations(case["y"])
        scores=control_chart(z,method,threshold,cooldown)
        alarms=[x["t"] for x in scores if x["alarm"]]
        change=case["change"]
        matches=[] if change is None else [t for t in alarms if change<=t<=change+tolerance]
        hit=bool(matches)
        tp+=int(hit)
        fn+=int(change is not None and not hit)
        fp+=len(alarms)-int(hit)
        if hit: delays.append(matches[0]-change)
        if case["kind"]=="none":
            null_n+=1
            null_alarms+=bool(alarms)
        detailed.append({"case_id":case["id"],"kind":case["kind"],"change":change,
                         "alarms":alarms,"detected":hit,
                         "delay":matches[0]-change if hit else None})
    precision=tp/max(tp+fp,1)
    recall=tp/max(tp+fn,1)
    result={"method":method,"threshold":threshold,"tp":tp,"fp":fp,"fn":fn,
            "precision":precision,"recall":recall,"f1":2*precision*recall/max(precision+recall,1e-12),
            "median_delay":float(np.median(delays)) if delays else None,
            "null_series_alarm_rate":null_alarms/max(null_n,1),"cases":len(cases)}
    return result,detailed
