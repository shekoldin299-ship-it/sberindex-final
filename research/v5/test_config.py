import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from model import load_config, forecast
import test_model

class ConfigTests(unittest.TestCase):
    def test_default_config_equivalent(self):
        fixture=test_model.CausalModelTests();fixture.setUp()
        for h in [1,3,6,12]:
            args=(fixture.y,fixture.c,fixture.n,'2024-06',h,fixture.b)
            np.testing.assert_array_equal(forecast(*args),forecast(*args,config=load_config()))

    def test_level_window_changes_prediction(self):
        fixture=test_model.CausalModelTests();fixture.setUp()
        config=load_config();config['level_window']=1
        args=(fixture.y,fixture.c,fixture.n,'2024-06',12,fixture.b)
        self.assertFalse(np.array_equal(forecast(*args),forecast(*args,config=config)))

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json'
            for key,value in [('level_window',0),('growth_clip',[2,1]),('pooled_growth_weight',2),('extra',1)]:
                config=load_config();config[key]=value;path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):load_config(path)

if __name__=='__main__':unittest.main()
