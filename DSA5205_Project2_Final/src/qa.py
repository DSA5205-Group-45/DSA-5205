"""Audit data, label timing, adjusted prices, and selected-bucket completeness."""
import numpy as np
import pandas as pd
from .pipeline import FEATURES

def audit_dataset(raw,screened,ready,audit):
    issues=[]
    if raw.duplicated(['date','ticker']).any():issues.append('duplicate raw date/ticker')
    if screened.duplicated(['decision_date','ticker']).any():issues.append('duplicate selected date/ticker')
    if ready.duplicated(['decision_date','ticker']).any():issues.append('duplicate ready date/ticker')
    if not np.isfinite(ready[FEATURES+['target_forward_return']].to_numpy(float)).all():issues.append('nonfinite model features or label')
    if not (pd.to_datetime(screened.decision_date)<pd.to_datetime(screened.execution_date)).all():issues.append('execution not after decision')
    if not (pd.to_datetime(screened.execution_date)<pd.to_datetime(screened.label_end_date)).all():issues.append('label end not after execution')
    counts=screened.groupby('decision_date').ticker.nunique()
    if (counts!=30).any():issues.append('screened date not 30 stocks')
    expected={'Low':3,'Medium':4,'High':3}
    for d,g in screened.groupby('decision_date'):
        for mg in ['High','Medium','Low']:
            for vg,n in expected.items():
                if len(g[(g.momentum_group==mg)&(g.volatility_group==vg)])!=n:
                    issues.append(f'bucket mismatch {d} {mg} {vg}')
    label_gap=(ready.target_forward_return-ready.target_spy_relative_return-ready.spy_return).abs().max()
    if pd.notna(label_gap) and label_gap>1e-9:issues.append('SPY label arithmetic mismatch')
    return {'pass':not issues,'issues':issues,'selected_dates':int(counts.size),'labeled_rows':len(ready),'missing_selected_labels':int(screened.target_forward_return.isna().sum()),'failed_screen_dates':int((audit.status!='ok').sum())}

def corporate_action_checks(raw):
    """Flag, do not overwrite, adjusted-vs-raw event days for manual spot checking."""
    g=raw.sort_values(['ticker','date']).copy()
    g['price_return']=g.groupby('ticker').close.pct_change(fill_method=None)
    g['adjusted_return']=g.groupby('ticker').adj_close.pct_change(fill_method=None)
    g['return_difference']=(g.adjusted_return-g.price_return).abs()
    flags=(g.stock_splits.fillna(0).ne(0)|g.dividends.fillna(0).ne(0)|g.return_difference.gt(.02))
    return g.loc[flags,['ticker','date','price_return','adjusted_return','return_difference','dividends','stock_splits']]
