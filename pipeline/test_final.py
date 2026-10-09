"""Behavioural checks for time isolation, publication-time joins and event scoring."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from forecasting import PooledRidge, features, fill_history, news_features, ranked_ids
from detection import control_chart, detect, evaluate_detector, innovations

ROOT=Path(__file__).parent


class FinalTests(unittest.TestCase):
    def test_release_warmup_and_h12_policy(self):
        from release_candidate import candidate_predict
        from benchmark import baseline_forecasts
        class FakeRidge:
            def predict(self,history,codes,dates,horizon):return np.ones(len(history))*123
        dates=pd.date_range("2023-01-01",periods=24,freq="MS")
        y=np.arange(1,25,dtype=float)[None,:]
        code=np.array([0])
        self.assertEqual(candidate_predict(FakeRidge(),y[:,:18],code,dates,6)[0],123)
        self.assertEqual(candidate_predict(FakeRidge(),y[:,:17],code,dates,6)[0],baseline_forecasts(y[0,:17],6)["seasonal_scaled"])
        self.assertEqual(candidate_predict(FakeRidge(),y,code,dates,12)[0],baseline_forecasts(y[0],12)["damped_trend"])

    def test_news_future_does_not_change_features(self):
        e=pd.read_csv(ROOT/"data/news_events.csv")
        now="2023-08-01"
        before=news_features(e,now)
        altered=e.copy()
        altered.loc[pd.to_datetime(altered.available_at)>=pd.Timestamp(now),["rate","delta_pp"]]=999
        np.testing.assert_equal(before,news_features(altered,now))
        self.assertEqual(before[0],8.5/25)

    def test_news_not_available_on_event_date(self):
        e=pd.read_csv(ROOT/"data/news_events.csv")
        self.assertEqual(news_features(e,"2023-08-15")[0],8.5/25)
        self.assertEqual(news_features(e,"2023-08-17")[0],12/25)

    def test_news_duplicates_rejected_in_integrity_check(self):
        e=pd.read_csv(ROOT/"data/news_events.csv")
        self.assertFalse(e.event_id.duplicated().any())
        self.assertTrue((pd.to_datetime(e.available_at)>pd.to_datetime(e.published_at,utc=True).dt.tz_localize(None)).all())

    def test_cohort_independent_of_future_coverage(self):
        idx=pd.MultiIndex.from_product([[1,2],list("abcdef")])
        a=pd.DataFrame(np.ones((12,24)),index=idx)
        first=ranked_ids(a,42)
        a.iloc[:,12:]=np.nan
        self.assertEqual(first,ranked_ids(a,42))

    def test_fill_uses_only_passed_history(self):
        a=np.array([[np.nan,2,np.nan,4],[1,np.nan,3,4.]])
        r=fill_history(a)
        np.testing.assert_equal(r[:,2],[2,3])
        self.assertTrue(np.isfinite(r).all())
        with self.assertRaises(ValueError):fill_history(np.full((1,3),np.nan))

    def test_ridge_fit_prefix_isolation(self):
        cfg=json.loads((ROOT/"final_config.json").read_text())
        rng=np.random.default_rng(15)
        y=np.exp(rng.normal(7,.1,(12,24)))
        codes=np.arange(12)%2
        dates=pd.date_range("2023-01-01",periods=24,freq="MS")
        a=PooledRidge(cfg,["a","b"]).fit(y[:,:15],codes,dates)
        p=a.predict(y[:,:15],codes,dates,3)
        y[:,15:]=1e12
        b=PooledRidge(cfg,["a","b"]).fit(y[:,:15],codes,dates)
        np.testing.assert_allclose(p,b.predict(y[:,:15],codes,dates,3),rtol=0,atol=0)
        self.assertTrue(np.isfinite(p).all() and (p>=0).all())

    def test_detector_prefix_invariance(self):
        y=np.exp(7+.01*np.arange(48))
        y[30:]*=1.5
        for m,t in [("cusum",6),("ewma",3),("window",3)]:
            full=detect(y,m,t)
            self.assertEqual(full[:29],detect(y[:29],m,t))

    def test_constant_no_alarms(self):
        for m in ["cusum","ewma","window"]:
            self.assertFalse(any(r["alarm"] for r in detect(np.ones(60)*100,m,3)))

    def test_prechange_alarm_is_false_positive(self):
        cases=[{"id":"x","kind":"up","change":20,"y":np.ones(30),
                "z":np.r_[np.zeros(15),10,np.zeros(14)]}]
        result,_=evaluate_detector(cases,"cusum",4)
        self.assertEqual(result["tp"],0)
        self.assertEqual(result["fp"],1)
        self.assertEqual(result["fn"],1)

    def test_persistent_standardized_shift_detected(self):
        z=np.r_[np.zeros(20),np.ones(10)*3]
        for m in ["cusum","ewma","window"]:
            alarms=[x["t"] for x in control_chart(z,m,3) if x["alarm"]]
            self.assertTrue(20<=alarms[0]<=22)

    def test_missing_detection_keeps_calendar_positions(self):
        y=np.ones(24)*100
        y[15]=np.nan
        scores=detect(y,"cusum",6)
        self.assertEqual([x["t"] for x in scores],list(range(24)))
        self.assertFalse(scores[15]["alarm"])


if __name__=="__main__":unittest.main()
