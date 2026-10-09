import unittest
import numpy as np,pandas as pd
from annual import annual_forecast
class AnnualTests(unittest.TestCase):
 def setUp(self):
  idx=pd.period_range('2018-01','2025-12',freq='M');t=np.arange(len(idx));v=np.exp(.01*t)*(1+.1*np.cos(t*np.pi/6));self.n=pd.DataFrame({k:v for k in ['food','nonfood','catering','services','total']},index=idx);self.y=np.array([[100.,120.,130.],[50.,55.,58.]]);self.c=['Все категории','Продовольствие']
 def test_short_history(self):
  a,_=annual_forecast(self.y,self.c,self.n,'2023-03');self.assertTrue(np.isfinite(a).all());self.assertEqual(a.shape,(2,))
 def test_future_invariance(self):
  a=annual_forecast(self.y,self.c,self.n,'2023-03')[0];n=self.n.copy();n.loc['2023-03':]=1e12;np.testing.assert_array_equal(a,annual_forecast(self.y,self.c,n,'2023-03')[0])
 def test_lag2(self):
  a=annual_forecast(self.y,self.c,self.n,'2023-03',2)[0];n=self.n.copy();n.loc['2023-02':]=1e12;np.testing.assert_array_equal(a,annual_forecast(self.y,self.c,n,'2023-03',2)[0])
 def test_too_short_rejected(self):
  with self.assertRaises(ValueError):annual_forecast(self.y[:,:2],self.c,self.n,'2023-03')
 def test_empty_series_rejected(self):
  y=self.y.copy();y[0]=np.nan
  with self.assertRaises(ValueError):annual_forecast(y,self.c,self.n,'2023-03')
 def test_exact_horizon12(self):
  # In a trend-only series, target lies one full year after origin.
  n=self.n.copy();t=np.arange(len(n));n.loc[:,:]=np.exp(.01*t)[:,None]*np.ones((1,5));y=np.array([[100.,100*np.exp(.01),100*np.exp(.02)]])
  p=annual_forecast(y,['Все категории'],n,'2023-03')[0];np.testing.assert_allclose(p,y[:,-1]*np.exp(.12),rtol=1e-12)
if __name__=='__main__':unittest.main()
