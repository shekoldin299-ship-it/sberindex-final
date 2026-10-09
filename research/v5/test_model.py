import unittest
import numpy as np,pandas as pd
from model import forecast

class CausalModelTests(unittest.TestCase):
    def setUp(self):
        idx=pd.period_range('2018-01','2025-12',freq='M');t=np.arange(len(idx))
        values=np.exp(.01*t)*(1+.1*np.cos(2*np.pi*t/12))
        self.n=pd.DataFrame({k:values*(i+1) for i,k in enumerate(['food','nonfood','catering','services','total'])},index=idx)
        self.y=np.array([np.arange(1,19)*10+100,np.arange(1,19)*7+80.])
        self.c=['Все категории','Продовольствие'];self.b=np.array([290.,220.])
    def test_national_future_cannot_change_forecast(self):
        for h in [1,3,6,12]:
            a=forecast(self.y,self.c,self.n,'2024-06',h,self.b)
            n=self.n.copy();n.loc['2024-06':]=1e12
            np.testing.assert_array_equal(a,forecast(self.y,self.c,n,'2024-06',h,self.b))
    def test_lag2_boundary(self):
        n=self.n.copy();n.loc['2024-05':]=1e12
        np.testing.assert_array_equal(forecast(self.y,self.c,self.n,'2024-06',1,self.b,2),forecast(self.y,self.c,n,'2024-06',1,self.b,2))
    def test_unchanged_medium_horizons(self):
        for h in [3,6]:np.testing.assert_array_equal(forecast(self.y,self.c,self.n,'2024-06',h,self.b),self.b)
    def test_h12_uses_external_level_not_baseline(self):
        np.testing.assert_array_equal(forecast(self.y,self.c,self.n,'2024-06',12,self.b),forecast(self.y,self.c,self.n,'2024-06',12,self.b*100))
    def test_reject_no_observations(self):
        y=self.y.copy();y[0]=np.nan
        with self.assertRaises(ValueError):forecast(y,self.c,self.n,'2024-06',1,self.b)
    def test_determinism_and_prefix_imputation(self):
        y=self.y.copy();y[0,:3]=np.nan;y[1,5]=np.nan
        a=forecast(y,self.c,self.n,'2024-06',1,self.b)
        np.testing.assert_array_equal(a,forecast(y,self.c,self.n,'2024-06',1,self.b))
        self.assertTrue(np.isfinite(a).all())
if __name__=='__main__':unittest.main()
