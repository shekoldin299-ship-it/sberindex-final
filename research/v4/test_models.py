import unittest
import numpy as np
from models import PanelModel, mixture_weights, MEMBERS, fill


class PrefixTests(unittest.TestCase):
    def test_detector_future_invariance(self):
        from detectors import change_scores
        rng=np.random.default_rng(12)
        y=np.exp(rng.normal(7,.1,24));f=np.full(24,7.)
        first=change_scores(y,f)[:,:18]
        y[18:]*=100;f[18:]+=8
        np.testing.assert_array_equal(first,change_scores(y,f)[:,:18])

    def test_boosted_model_deterministic_prefix(self):
        rng=np.random.default_rng(4);y=np.exp(rng.normal(6,.1,(60,18)))
        codes=np.arange(60)%6
        p=PanelModel(y[:,:12],codes).predict(3)['factor_boost']
        y[:,12:]*=100
        q=PanelModel(y[:,:12],codes).predict(3)['factor_boost']
        np.testing.assert_array_equal(p,q)

    def test_future_cannot_change_predictions(self):
        rng=np.random.default_rng(7)
        all_values=np.exp(rng.normal(5,.1,(120,24)))
        codes=np.arange(120)%6
        p=PanelModel(all_values[:,:18],codes,boost=False).predict(6)
        all_values[:,18:]*=10000
        q=PanelModel(all_values[:,:18],codes,boost=False).predict(6)
        for name in p:np.testing.assert_array_equal(p[name],q[name])

    def test_unmatured_losses_do_not_change_weights(self):
        r={'category':'x','horizon':3,'target_index':20,**{n:1+i for i,n in enumerate(MEMBERS)}}
        np.testing.assert_array_equal(mixture_weights([r]*10,'x',3,18),np.ones(5)/5)

    def test_constant_rows_and_factor_scale(self):
        y=np.full((60,12),1000.)
        p=PanelModel(y,np.arange(60)%6,boost=False).predict(12)
        for v in p.values():np.testing.assert_allclose(v,1000,atol=1e-8)

    def test_invalid_history_rejected(self):
        with self.assertRaises(ValueError):fill([[np.nan,np.nan]])


if __name__=='__main__':unittest.main()
