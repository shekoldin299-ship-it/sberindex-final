"""Contracts for independently implemented V3 methods."""
import unittest
import numpy as np
import pandas as pd
from upgrade_v3 import panel_candidates, select_policy, panel_alarms, factor_scores

class V3Contracts(unittest.TestCase):
    def test_exact_seasonal_panel(self):
        # Zero-growth, repeating seasonal histories must reproduce the next season.
        year=np.arange(100.,112.)
        a=np.stack([np.tile(year,2)[:18],np.tile((year+1)*2-1,2)[:18]])
        for p in panel_candidates(a,np.array([0,0]),3).values():
            np.testing.assert_allclose(p,[108.,217.],rtol=1e-5)

    def test_missing_series_does_not_create_fake_zero(self):
        a=np.stack([np.full(18,100.),np.full(18,np.nan)])
        for p in panel_candidates(a,np.array([0,0]),1).values():
            self.assertAlmostEqual(p[0],100.)
            self.assertTrue(np.isnan(p[1]))

    def test_category_isolation(self):
        a=np.full((4,18),100.)
        b=a.copy();b[2:]*=50
        codes=np.array([0,0,1,1])
        for name,p in panel_candidates(a,codes,1).items():
            np.testing.assert_allclose(p[:2],panel_candidates(b,codes,1)[name][:2])

    def test_policy_preserves_diagnostic_and_small_gain(self):
        rows=[]
        names=[f'panel_k{k}_w{w:g}' for k in (3,6) for w in (.25,.75)]+['v2_panel_k3','v2_panel_k6']
        for h in (1,12):
            rows.append({'horizon':h,'category':'x','actual':100.,'v2':110.,**{name:109.9 for name in names}})
        p=select_policy(pd.DataFrame(rows))
        self.assertEqual(p['1|x']['model'],'v2');self.assertEqual(p['12|x']['model'],'v2')

    def test_detection_is_prefix_causal(self):
        y=np.full(24,100.);y[17:]=125.
        a=factor_scores(y,np.zeros(12),np.full(12,.02))
        y[21:]=1e6
        b=factor_scores(y,np.zeros(12),np.full(12,.02))
        np.testing.assert_equal(a[:21],b[:21])
        self.assertEqual([t for t in panel_alarms(a,2,4) if t<21],[t for t in panel_alarms(b,2,4) if t<21])
        self.assertTrue(any(17<=t<=19 for t in panel_alarms(a,2,4)))

if __name__=='__main__':unittest.main()
