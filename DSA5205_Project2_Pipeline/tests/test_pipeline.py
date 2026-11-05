import numpy as np
import pandas as pd
from src.pipeline import _choose, decision_dates
from src.backtest import portfolio_backtest

def test_selection_rule():
    cell=pd.DataFrame({'ticker':['A','B','C','D'],'mom_2_12':[0.1,0.4,0.2,0.3]})
    assert _choose(cell,1,'High').ticker.iloc[0]=='B'
    assert _choose(cell,1,'Low').ticker.iloc[0]=='A'

def test_calendar_frequency():
    days=pd.bdate_range('2024-01-01','2024-12-31')
    panel=pd.DataFrame({'date':days,'ticker':'SPY'})
    base={'research_start':'2024-01-01','research_end':'2024-12-31'}
    assert len(decision_dates(panel,dict(base,rebalance_frequency='monthly')))==12
    assert len(decision_dates(panel,dict(base,rebalance_frequency='quarterly')))==4
    assert len(decision_dates(panel,dict(base,rebalance_frequency='semiannual')))==2

def test_cost_and_differential_trading():
    rows=[]
    for d in pd.to_datetime(['2024-01-31','2024-02-29']):
        for t,r,m in [('A',0.10,0.5),('B',0.0,0.1)]:
            rows.append({'decision_date':d,'ticker':t,'prediction':m,'mom_2_12':m,'target_forward_return':r,'spy_return':0.01,'qqq_return':0.02})
    p=pd.DataFrame(rows)
    returns,trades,holdings=portfolio_backtest(p,'monthly',20,top_n=2)
    first=returns[(returns.strategy=='model_top10')].iloc[0]
    assert np.isclose(first.cost,0.002)
    second=returns[(returns.strategy=='model_top10')].iloc[1]
    assert second.turnover_two_way<0.1  # drifted weights only, not full liquidation
