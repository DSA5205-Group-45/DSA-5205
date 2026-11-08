import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

FEATURES=['mom_21','mom_63','mom_126','mom_2_12','relative_mom_63','vol_21','vol_63','beta_252','drawdown_63','ma_ratio_50','ma_ratio_200','rsi_14','ln_adv_20','volume_ratio_21','range_hl']

def pools(top):
    top=top.sort_values(['mom_2_12','ticker'],ascending=[False,True]).reset_index(drop=True)
    if len(top)<90: raise ValueError('Insufficient universe')
    sixty=[];thirty=[]
    for group,idx in zip(['High','Medium','Low'],np.array_split(np.arange(len(top)),3)):
        g=top.iloc[idx].copy()
        if group=='High': pick=g.sort_values('mom_2_12',ascending=False).head(20)
        elif group=='Low': pick=g.sort_values('mom_2_12').head(20)
        else: pick=g.assign(distance=(g.mom_2_12-g.mom_2_12.median()).abs()).sort_values(['distance','ticker']).head(20)
        sixty.append(pick.assign(momentum_group=group))
        g=g.sort_values(['vol_63','ticker']).reset_index(drop=True)
        for vg,ix,n in zip(['Low','Medium','High'],np.array_split(np.arange(len(g)),3),[3,4,3]):
            c=g.iloc[ix]
            if group=='High': p=c.sort_values('mom_2_12',ascending=False).head(n)
            elif group=='Low': p=c.sort_values('mom_2_12').head(n)
            else: p=c.assign(distance=(c.mom_2_12-c.mom_2_12.median()).abs()).sort_values(['distance','ticker']).head(n)
            thirty.append(p.assign(momentum_group=group,volatility_group=vg))
    s=pd.concat(sixty);t=pd.concat(thirty)
    assert s.ticker.nunique()==60 and t.ticker.nunique()==30
    return s,t

def capped(pred,cap=.2):
    x=np.maximum(np.asarray(pred,float),0)
    if x.sum()==0:x=np.ones(len(x))
    w=np.zeros(len(x));active=np.ones(len(x),bool)
    while active.any():
        a=x[active]
        if a.sum()==0:a=np.ones(len(a))
        proposal=(1-w.sum())*a/a.sum();ids=np.where(active)[0];over=proposal>cap+1e-12
        if not over.any():w[ids]=proposal;break
        w[ids[over]]=cap;active[ids[over]]=False
    assert np.isclose(w.sum(),1) and w.max()<=cap+1e-9
    return w

def train_ridge(s60,train_end='2020-01-01',val_end='2022-01-01'):
    tr=s60[(s60.decision_date<train_end)&(s60.label_end_date<train_end)].dropna(subset=['target_forward_return'])
    va=s60[(s60.decision_date>=train_end)&(s60.decision_date<val_end)&(s60.label_end_date<val_end)].dropna(subset=['target_forward_return'])
    if tr.empty or va.empty:raise ValueError('Empty train or validation')
    records=[]
    for alpha in [.1,1,10,100,1000,10000]:
        m=make_pipeline(StandardScaler(),Ridge(alpha=alpha));m.fit(tr[FEATURES],tr.target_forward_return)
        records.append({'alpha':alpha,'train_mse':mean_squared_error(tr.target_forward_return,m.predict(tr[FEATURES])),'validation_mse':mean_squared_error(va.target_forward_return,m.predict(va[FEATURES]))})
    grid=pd.DataFrame(records).sort_values(['validation_mse','alpha']);best=float(grid.iloc[0].alpha)
    fit=s60[(s60.decision_date<val_end)&(s60.label_end_date<val_end)].dropna(subset=['target_forward_return'])
    model=make_pipeline(StandardScaler(),Ridge(alpha=best));model.fit(fit[FEATURES],fit.target_forward_return)
    return model,grid,{'alpha':best,'train_rows':len(tr),'validation_rows':len(va),'refit_rows':len(fit)}

def targets_for_date(top,s60,s30,model):
    top=top.sort_values(['mom_2_12','ticker'],ascending=[False,True]);b10=top.head(10);b30=top.head(30)
    a=s60.copy();a['prediction']=model.predict(a[FEATURES]);b=s30.copy();b['prediction']=model.predict(b[FEATURES])
    free=a.sort_values(['prediction','ticker'],ascending=[False,True]).head(10)
    quota=pd.concat([a[a.momentum_group==g].sort_values(['prediction','ticker'],ascending=[False,True]).head(n) for g,n in [('High',3),('Medium',4),('Low',3)]])
    rigid=b.sort_values(['prediction','ticker'],ascending=[False,True]).head(10)
    selections={'B1_Momentum10':(b10,np.ones(10)/10),'B2_Momentum30':(b30,np.ones(30)/30),'B3_MomentumDoubleSort10':(s30.sort_values(['mom_2_12','ticker'],ascending=[False,True]).head(10),np.ones(10)/10),'R1_RidgeWeighted10':(b10,capped(model.predict(b10[FEATURES]))),'R2_RidgeFree60':(free,np.ones(10)/10),'R3_RidgeQuota343':(quota,np.ones(10)/10),'R4_RidgeDoubleSort30':(rigid,np.ones(10)/10)}
    return {name:dict(zip(g.ticker,w)) for name,(g,w) in selections.items()}

def simulate(selections,returns_lookup,cost_bps=20):
    rows=[];trades=[];holdings=[]
    for name in next(iter(selections.values())):
        old={};old_returns={}
        for date,groups in sorted(selections.items()):
            target=groups[name];grown={t:w*(1+old_returns[t]) for t,w in old.items()};denom=sum(grown.values());pre={t:v/denom for t,v in grown.items()} if denom>0 else {}
            r={t:returns_lookup[(date,t)] for t in target}
            if any(not np.isfinite(v) for v in r.values()):raise ValueError(f'Missing return {name} {date}')
            turnover=sum(abs(target.get(t,0)-pre.get(t,0)) for t in set(target)|set(pre));cost=turnover*cost_bps/10000;gross=sum(w*r[t] for t,w in target.items())
            rows.append({'strategy':name,'decision_date':date,'gross_return':gross,'cost':cost,'net_return':gross-cost,'turnover_two_way':turnover})
            for t in set(target)|set(pre):
                delta=target.get(t,0)-pre.get(t,0)
                action='BUY' if pre.get(t,0)==0 else 'EXIT' if target.get(t,0)==0 else 'ADD' if delta>1e-10 else 'REDUCE' if delta< -1e-10 else 'HOLD'
                trades.append({'strategy':name,'decision_date':date,'ticker':t,'action':action,'pretrade_weight':pre.get(t,0),'target_weight':target.get(t,0),'delta_weight':delta})
            for t,w in target.items():holdings.append({'strategy':name,'decision_date':date,'ticker':t,'target_weight':w,'stock_period_return':r[t],'gross_contribution':w*r[t]})
            old=target;old_returns=r
    return pd.DataFrame(rows),pd.DataFrame(trades),pd.DataFrame(holdings)

def performance(df,frequency):
    n={'monthly':12,'quarterly':4,'semiannual':2}[frequency];rows=[]
    for name,g in df.groupby('strategy'):
        g=g.sort_values('decision_date');r=g.net_return;wealth=(1+r).cumprod();std=r.std()
        rows.append({'strategy':name,'periods':len(r),'annualized_return':wealth.iloc[-1]**(n/len(r))-1,'annualized_volatility':std*np.sqrt(n),'sharpe_rf0':r.mean()/std*np.sqrt(n) if std>0 else np.nan,'max_drawdown':(wealth/wealth.cummax()-1).min(),'avg_two_way_turnover':g.turnover_two_way.mean()})
    return pd.DataFrame(rows).sort_values('annualized_return',ascending=False)
