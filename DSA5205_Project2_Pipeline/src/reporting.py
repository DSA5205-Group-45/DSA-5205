"""Course §4.9 reporting figures and sensitivity tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from .backtest import portfolio_backtest, performance

def make_reports(predictions, frequency, output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    actual=predictions.target_forward_return.to_numpy()
    pred=predictions.prediction.to_numpy()
    # Out-of-sample R2 relative to historical training-mean naive forecast
    # must be computed separately from test labels: caller can provide training mean.
    r2=float(r2_score(actual,pred))
    (output_dir/'test_oos_r2.txt').write_text(f'Test R2 relative to test sample mean (descriptive, not historical-mean baseline): {r2:.6f}\n')
    periods,trade,hold=portfolio_backtest(predictions,frequency,20)
    fig,ax=plt.subplots(figsize=(10,4))
    for strategy,g in periods.groupby('strategy'):
        g=g.sort_values('decision_date')
        ax.plot(g.decision_date,(1+g.net_return).cumprod(),label=strategy)
    ax.legend();ax.set_title('OOS cumulative net wealth');fig.tight_layout();fig.savefig(output_dir/'cumulative_wealth.png',dpi=170);plt.close(fig)
    annual={'monthly':12,'quarterly':4,'semiannual':2}[frequency]
    sub=[]
    for (name,year),g in periods.assign(year=lambda x:pd.to_datetime(x.decision_date).dt.year).groupby(['strategy','year']):
        r=g.net_return
        sub.append({'strategy':name,'year':year,'periods':len(r),'annualized_return':float((1+r).prod()**(annual/len(r))-1),'sharpe_rf0':float(r.mean()/r.std(ddof=1)*np.sqrt(annual)) if len(r)>1 and r.std(ddof=1)>0 else np.nan})
    sub=pd.DataFrame(sub);sub.to_csv(output_dir/'yearly_return_sharpe.csv',index=False)
    for col,title in [('annualized_return','Annualized return by year'),('sharpe_rf0','Sharpe by year (rf=0)')]:
        pivot=sub.pivot(index='year',columns='strategy',values=col)
        ax=pivot.plot(kind='bar',figsize=(10,4),title=title)
        ax.figure.tight_layout();ax.figure.savefig(output_dir/f'yearly_{col}.png',dpi=170);plt.close(ax.figure)
    costs=[]
    for bps in [5,10,20,40]:
        r,_,_=portfolio_backtest(predictions,frequency,bps)
        summary=performance(r,frequency)
        summary['cost_bps_per_side']=bps
        costs.append(summary)
    pd.concat(costs,ignore_index=True).to_csv(output_dir/'cost_sensitivity.csv',index=False)
    ic=[]
    for date,g in predictions.groupby('decision_date'):
        ic.append({'decision_date':date,'rank_ic':g.prediction.corr(g.target_forward_return,method='spearman')})
    pd.DataFrame(ic).to_csv(output_dir/'monthly_rank_ic.csv',index=False)
    return sub
