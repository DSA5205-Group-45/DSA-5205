import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from strategies import pools,targets_for_date,capped,FEATURES

class Dummy:
    def predict(self, X):
        return X["mom_2_12"].to_numpy()

def test_original_baseline_from_3x3():
    n=300
    top=pd.DataFrame({"ticker":[f"T{i:03d}" for i in range(n)],"mom_2_12":np.linspace(3,-3,n),"vol_63":np.tile(np.linspace(.1,.9,100),3)})
    for f in FEATURES:
        if f not in top.columns:top[f]=0.2
    s60,s30=pools(top)
    results=targets_for_date(top,s60,s30,Dummy())
    expected=set(s30.sort_values(["mom_2_12","ticker"],ascending=[False,True]).head(10).ticker)
    assert set(results["B3_MomentumDoubleSort10"])==expected
    assert len(results)==7
    assert len(results["B3_MomentumDoubleSort10"])==10
    assert np.isclose(sum(results["B3_MomentumDoubleSort10"].values()),1)

def test_weight_cap():
    w=capped([.9,.5,.2,.1,.1,.1,.1,.1,.1,.1])
    assert np.isclose(w.sum(),1) and max(w)<=.20000001
