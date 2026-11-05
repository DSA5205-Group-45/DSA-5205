import argparse
from src.models import run_model
from src.backtest import portfolio_backtest,performance
from src.pipeline import ROOT
from src.reporting import make_reports
parser=argparse.ArgumentParser()
parser.add_argument('--frequency',choices=['monthly','quarterly','semiannual'],default='monthly')
parser.add_argument('--model',choices=['ridge','lasso','rf','xgb'],default='ridge')
args=parser.parse_args()
pred,grid,estimator=run_model(args.frequency,args.model)
returns,trades,holdings=portfolio_backtest(pred,args.frequency)
out=ROOT/'outputs'/'models'/args.frequency/args.model
returns.to_csv(out/'portfolio_returns.csv',index=False)
trades.to_csv(out/'rebalance_trade_log.csv',index=False)
holdings.to_csv(out/'monthly_stock_pnl.csv',index=False)
performance(returns,args.frequency).to_csv(out/'performance_summary.csv',index=False)
print(performance(returns,args.frequency).to_string(index=False))

make_reports(pred,args.frequency,out)
