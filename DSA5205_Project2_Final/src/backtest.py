"""Weight-level monthly/quarterly/semiannual strategy accounting, net of turnover costs.
Assumes fractional shares, no cash drag, instant target weights at execution open.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .pipeline import ROOT

def portfolio_backtest(predictions, frequency='monthly', cost_bps=20, top_n=10):
    p=predictions.sort_values(['decision_date','ticker']).copy()
    p['decision_date']=pd.to_datetime(p.decision_date)
    strategies=['model_top10','momentum_top10','equal_weight_30']
    results=[]; trades=[]; holdings=[]
    for strategy in strategies:
        prev_weights={};prev_returns={}
        for date,g in p.groupby('decision_date',sort=True):
            if strategy=='model_top10': picks=g.sort_values(['prediction','ticker'],ascending=[False,True]).head(top_n)
            elif strategy=='momentum_top10': picks=g.sort_values(['mom_2_12','ticker'],ascending=[False,True]).head(top_n)
            else: picks=g
            target={t:1/len(picks) for t in picks.ticker} if len(picks) else {}
            # Drift previous weights using the prior period's actual returns.
            grown={t:w*(1+prev_returns[t]) for t,w in prev_weights.items()}
            denominator=sum(grown.values())
            pre={t:v/denominator for t,v in grown.items()} if denominator>0 else {}
            tickers=sorted(set(pre)|set(target))
            turnover=sum(abs(target.get(t,0)-pre.get(t,0)) for t in tickers)
            cost=turnover*cost_bps/10000
            realized=dict(zip(g.ticker,g.target_forward_return))
            gross=sum(w*realized[t] for t,w in target.items())
            net=gross-cost
            for t in tickers:
                old=pre.get(t,0);new=target.get(t,0);delta=new-old
                action=('BUY' if old==0 and new>0 else 'EXIT' if new==0 else 'ADD' if delta>1e-12 else 'REDUCE' if delta< -1e-12 else 'HOLD')
                trades.append({'strategy':strategy,'decision_date':date,'ticker':t,'pretrade_weight':old,'target_weight':new,'delta_weight':delta,'action':action,'cost_contribution':abs(delta)*cost_bps/10000})
            for t,w in target.items():
                r=realized[t]
                holdings.append({'strategy':strategy,'decision_date':date,'ticker':t,'target_weight':w,'period_stock_return':r,'gross_return_contribution':w*r,'profit_status':'PROFIT' if r>0 else 'LOSS' if r<0 else 'FLAT'})
            results.append({'strategy':strategy,'decision_date':date,'gross_return':gross,'turnover_two_way':turnover,'cost':cost,'net_return':net})
            prev_weights=target;prev_returns={t:realized[t] for t in target}
    # Benchmarks: same aligned execution-to-execution periods, gross ETF buy-and-hold.
    # An initial one-way entry fee is applied once, not a monthly rebalance fee.
    for ticker,col in [('SPY','spy_return'),('QQQ','qqq_return')]:
        for i,(date,g) in enumerate(p.groupby('decision_date',sort=True)):
            vals=g[col].dropna().unique()
            if len(vals)!=1:raise ValueError(f'Benchmark {ticker} inconsistent on {date}')
            fee=cost_bps/10000 if i==0 else 0
            results.append({'strategy':ticker,'decision_date':date,'gross_return':float(vals[0]),'turnover_two_way':1.0 if i==0 else 0.0,'cost':fee,'net_return':float(vals[0])-fee})
    return pd.DataFrame(results),pd.DataFrame(trades),pd.DataFrame(holdings)

def performance(returns, frequency='monthly'):
    annual={'monthly':12,'quarterly':4,'semiannual':2}[frequency]
    rows=[]
    for name,g in returns.groupby('strategy'):
        r=g.sort_values('decision_date').net_return.astype(float)
        wealth=(1+r).cumprod()
        rows.append({'strategy':name,'periods':len(r),'annualized_return':float(wealth.iloc[-1]**(annual/len(r))-1),'annualized_volatility':float(r.std(ddof=1)*np.sqrt(annual)),'sharpe_rf0':float(r.mean()/r.std(ddof=1)*np.sqrt(annual)) if r.std(ddof=1)>0 else np.nan,'max_drawdown':float((wealth/wealth.cummax()-1).min()),'mean_two_way_turnover':float(g.turnover_two_way.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Additions for the four-model comparison.
#
# The functions above are copied verbatim from the authoritative pipeline and
# define the trading rules: prediction-ranked Top-10, equal weights, drifted
# pre-trade weights, turnover = sum(|target - pretrade|), 20 bps per traded
# side, next-open execution encoded in ``target_forward_return``.  The helpers
# below only *call* that implementation so all four models share exactly the
# same portfolio construction.
# ---------------------------------------------------------------------------
BENCHMARK_STRATEGIES = ("SPY", "QQQ")


def run_single_strategy_backtest(
    predictions,
    model_name,
    frequency="monthly",
    cost_bps=20,
    top_n=10,
):
    """Backtest one model with the original rules and label its rows."""
    required = {"decision_date", "ticker", "prediction", "mom_2_12", "target_forward_return", "spy_return", "qqq_return"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns for backtest: {sorted(missing)}")
    returns, trades, holdings = portfolio_backtest(predictions, frequency, cost_bps, top_n)
    keep = returns.loc[returns.strategy.isin(["model_top10", *BENCHMARK_STRATEGIES])].copy()
    keep["strategy"] = keep["strategy"].replace({"model_top10": model_name})
    trades = trades.loc[trades.strategy == "model_top10"].copy()
    trades["strategy"] = model_name
    holdings = holdings.loc[holdings.strategy == "model_top10"].copy()
    holdings["strategy"] = model_name
    return keep, trades, holdings


def combine_model_backtests(results, frequency="monthly", cost_bps=20):
    """Concatenate per-model backtests, keeping one copy of each benchmark."""
    frames = []
    benchmark_rows = None
    for model_name, predictions in results.items():
        returns, _, _ = run_single_strategy_backtest(
            predictions, model_name, frequency=frequency, cost_bps=cost_bps
        )
        frames.append(returns.loc[returns.strategy == model_name])
        if benchmark_rows is None:
            benchmark_rows = returns.loc[returns.strategy.isin(BENCHMARK_STRATEGIES)].copy()
    frames.append(benchmark_rows)
    combined = pd.concat(frames, ignore_index=True)
    combined["decision_date"] = pd.to_datetime(combined.decision_date)
    return combined.sort_values(["strategy", "decision_date"]).reset_index(drop=True)


def reconcile_holdings(returns, holdings):
    """Verify that weighted stock contributions sum to the portfolio return."""
    if holdings.empty:
        return 0.0
    contributions = holdings.groupby(["strategy", "decision_date"]).gross_return_contribution.sum()
    gross = returns.set_index(["strategy", "decision_date"]).gross_return
    aligned = pd.concat([contributions.rename("contrib"), gross.rename("gross")], axis=1, join="inner")
    if aligned.empty:
        return float("nan")
    return float((aligned.contrib - aligned.gross).abs().max())


def assert_execution_ordering(frame):
    """Decision strictly before execution strictly before label settlement."""
    decision = pd.to_datetime(frame.decision_date)
    execution = pd.to_datetime(frame.execution_date)
    label_end = pd.to_datetime(frame.label_end_date)
    if not (decision < execution).all():
        raise AssertionError("A trade was executed on or before its decision date")
    if not (execution < label_end).all():
        raise AssertionError("A label ends on or before its execution date")
    return True
