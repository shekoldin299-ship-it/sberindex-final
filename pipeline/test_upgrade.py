import unittest
import numpy as np
import pandas as pd
from upgrade_v2 import candidates,choose_policy,apply_policy,quantile,robust_chart

class UpgradeTests(unittest.TestCase):
    def test_quantile_finite_sample_rank(self):
        self.assertEqual(quantile(np.arange(19)),17)
    def test_quantile_insufficient_sample(self):
        with self.assertRaises(ValueError):quantile([1,2])
    def test_fractional_model_name_and_h12(self):
        w=pd.DataFrame({'horizon':[1,12],'category':['x','x'],'blend_0.25':[20,21],'damped_trend':[30,31]})
        np.testing.assert_equal(apply_policy(w,{'1|x':{'model':'blend_0.25'}}),[20,31])
    def test_small_gain_rejected(self):
        w=candidates(pd.DataFrame({'horizon':[1],'category':['x'],'actual':[100.], 'pooled_ridge':[109.9],'seasonal_scaled':[110.],'chronos_adapted':[120.]}))
        self.assertEqual(choose_policy(w)['1|x']['model'],'blend_0')
    def test_large_gain_accepted(self):
        w=candidates(pd.DataFrame({'horizon':[1],'category':['x'],'actual':[100.], 'pooled_ridge':[100.],'seasonal_scaled':[110.],'chronos_adapted':[120.]}))
        self.assertEqual(choose_policy(w)['1|x']['model'],'blend_1')
    def test_online_prefix_invariance(self):
        z=np.array([np.nan]*12+[1.,3.,3.,-2.,-3.,2.,1.,0.])
        for method in ('cusum_v1','median3','multiscale'):
            a=robust_chart(z,method,3)
            self.assertEqual(a[:17],robust_chart(z[:17],method,3))
    def test_median_rejects_isolated_outlier(self):
        self.assertFalse(any(r['alarm'] for r in robust_chart([0,0,12,0,0],'median3',3)))
    def test_missing_never_alarm(self):
        self.assertTrue(all(not r['alarm'] for r in robust_chart([np.nan]*24,'multiscale',2)))

if __name__=='__main__':unittest.main()
