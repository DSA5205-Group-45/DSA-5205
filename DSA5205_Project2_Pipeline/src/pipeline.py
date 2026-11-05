from pathlib import Path
import json, time, math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['mom_21','mom_63','mom_126','mom_2_12','relative_mom_63','vol_21','vol_63','beta_252','drawdown_63','ma_ratio_50','ma_ratio_200','rsi_14','ln_adv_20','volume_ratio_21','range_hl']

def get_universe(max_tickers=None):
    # Wikipedia's LIVE constituents: intentionally NOT historical point-in-time membership.
    url='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    import requests
    from io import StringIO
    headers={'User-Agent':'Mozilla/5.0 (educational research; contact via Wikipedia)'}
    response=requests.get(url,headers=headers,timeout=30);response.raise_for_status()
    table=pd.read_html(StringIO(response.text),match='Symbol')[0]
    if 'Symbol' not in table: raise ValueError('Wikipedia table layout changed')
    symbols=table['Symbol'].astype(str).str.strip().str.replace('.','-',regex=False).tolist()
    symbols=list(dict.fromkeys(s for s in symbols if s and s!='nan'))
    if max_tickers: symbols=symbols[:int(max_tickers)]
    if len(symbols)<300 and not max_tickers: raise RuntimeError('Too few symbols; inspect source')
    return symbols

def _normalize(frame,ticker):
    if frame is None or frame.empty: return pd.DataFrame()
    frame=frame.copy()
    if isinstance(frame.columns,pd.MultiIndex): frame.columns=frame.columns.get_level_values(0)
    frame=frame.reset_index()
    if 'Date' not in frame and 'Datetime' in frame: frame=frame.rename(columns={'Datetime':'Date'})
    mapping={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume','Dividends':'dividends','Stock Splits':'stock_splits'}
    frame=frame.rename(columns=mapping)
    needed=['date','open','high','low','close','adj_close','volume']
    if any(c not in frame for c in needed): raise ValueError(f'{ticker}: missing required OHLCV columns')
    frame['date']=pd.to_datetime(frame['date']).dt.tz_localize(None).dt.normalize()
    frame['ticker']=ticker
    for c in ['dividends','stock_splits']:
        if c not in frame:frame[c]=np.nan
    return frame[['date','ticker','open','high','low','close','adj_close','volume','dividends','stock_splits']].drop_duplicates(['date','ticker']).sort_values('date')

def download(cfg, force=False):
    import yfinance as yf
    rawdir=ROOT/'data/raw';rawdir.mkdir(parents=True,exist_ok=True)
    symbols=get_universe(cfg.get('max_tickers'))
    symbols=sorted(set(symbols+['SPY','QQQ']))
    (rawdir/'ticker_universe.json').write_text(json.dumps({'universe':'Wikipedia current S&P 500 constituents (survivorship-biased)','retrieved_utc':pd.Timestamp.utcnow().isoformat(),'tickers':symbols},indent=2))
    frames=[];status=[]
    for n,ticker in enumerate(symbols,1):
        state='not_downloaded'
        path=rawdir/'cache'/f'{ticker}.parquet';path.parent.mkdir(exist_ok=True)
        try:
            if path.exists() and not force:
                part=pd.read_parquet(path);state='cached'
            else:
                last_error=None
                for attempt in range(cfg.get('download_retries',3)):
                    try:
                        result=yf.download(ticker,start=cfg['raw_start'],end=cfg['download_end_exclusive'],auto_adjust=False,actions=True,progress=False,threads=False,keepna=True,repair=False,multi_level_index=False,timeout=25)
                        part=_normalize(result,ticker)
                        if part.empty: raise ValueError('empty response')
                        break
                    except Exception as exc:
                        last_error=exc
                        if attempt+1==cfg.get('download_retries',3): raise
                        time.sleep(2*(attempt+1))
                part.to_parquet(path,index=False);state='downloaded'
            frames.append(part);status.append({'ticker':ticker,'status':state,'rows':len(part),'first_date':str(part.date.min().date()),'last_date':str(part.date.max().date())})
        except Exception as exc:
            status.append({'ticker':ticker,'status':'failed','rows':0,'error':str(exc)[:250]})
        if n%25==0:print(f'Downloaded/cached {n}/{len(symbols)}')
        if cfg.get('request_pause_seconds',0)>0 and state=='downloaded':time.sleep(cfg['request_pause_seconds'])
    log=pd.DataFrame(status);log.to_csv(rawdir/'download_status.csv',index=False)
    if not frames:raise RuntimeError('No data downloaded. Check internet/yfinance.')
    raw=pd.concat(frames,ignore_index=True).drop_duplicates(['date','ticker']).sort_values(['ticker','date'])
    raw.to_parquet(rawdir/'market_data.parquet',index=False)
    for benchmark in ['SPY','QQQ']:
        if benchmark not in raw.ticker.unique(): raise RuntimeError(f'Mandatory benchmark {benchmark} failed to download')
    failures=log.loc[log.status=='failed']
    summary={'requested':len(symbols),'failed':len(failures),'failure_rate':round(len(failures)/len(symbols),4),'failed_tickers':failures.ticker.tolist(),'universe_type':'current_constituents_backfilled','source':'Wikipedia + Yahoo Finance'}
    (rawdir/'download_audit.json').write_text(json.dumps(summary,indent=2))
    if summary['failure_rate']>0.10: raise RuntimeError('More than 10% ticker downloads failed; see download_audit.json')
    return raw,log

def _rsi(series,window=14):
    delta=series.diff();up=delta.clip(lower=0);down=-delta.clip(upper=0)
    avg_up=up.rolling(window,min_periods=window).mean();avg_down=down.rolling(window,min_periods=window).mean()
    result=100-100/(1+avg_up/avg_down.replace(0,np.nan))
    result=result.mask((avg_down==0)&(avg_up>0),100).mask((avg_down==0)&(avg_up==0),50)
    return result

def features(raw,cfg):
    # Prices are not globally forward-filled. Compute from genuine observations; limited feature ffill only later.
    spy=raw.loc[raw.ticker=='SPY',['date','adj_close']].drop_duplicates('date').sort_values('date')
    spy['spy_ret']=spy.adj_close.pct_change(fill_method=None)
    spy['spy_mom_63']=spy.adj_close/spy.adj_close.shift(63)-1
    results=[]
    for ticker,g in raw.groupby('ticker',sort=False):
        # Align to the common SPY trading calendar so missing sessions remain visible.
        # Do not manufacture pre-IPO or post-delisting prices.
        g=spy[['date','spy_ret','spy_mom_63']].merge(g.sort_values('date'),on='date',how='left')
        g['ticker']=ticker
        p=g.adj_close.where(g.adj_close>0);r=p.pct_change(fill_method=None)
        for w in (21,63,126):g[f'mom_{w}']=p/p.shift(w)-1
        g['vol_21']=r.rolling(21,min_periods=20).std()*np.sqrt(252)
        g['vol_63']=r.rolling(63,min_periods=60).std()*np.sqrt(252)
        paired=r.notna()&g.spy_ret.notna()
        cov=r.where(paired).rolling(252,min_periods=240).cov(g.spy_ret.where(paired))
        var=g.spy_ret.where(paired).rolling(252,min_periods=240).var()
        g['beta_252']=cov/var.replace(0,np.nan)
        for w in (50,200):g[f'ma_ratio_{w}']=p/p.rolling(w,min_periods=math.ceil(.95*w)).mean()-1
        g['drawdown_63']=p/p.rolling(63,min_periods=60).max()-1
        g['rsi_14']=_rsi(p)
        dv=g.close*g.volume
        g['adv_20']=dv.rolling(20,min_periods=19).mean()
        g['ln_adv_20']=np.log1p(g.adv_20.clip(lower=0))
        g['volume_ratio_21']=g.volume/g.volume.rolling(21,min_periods=20).mean().replace(0,np.nan)
        g['range_hl']=(g.high-g.low)/g.close.replace(0,np.nan)
        g['relative_mom_63']=g.mom_63-g.spy_mom_63
        # MOM2-12 uses actual monthly observations, not a 21/252-day approximation.
        monthly=g.set_index('date').adj_close.resample('ME').last()
        mom=monthly.shift(1)/monthly.shift(12)-1
        g['mom_2_12']=g.date.dt.to_period('M').map(mom.rename(index=lambda x:x.to_period('M')))
        g['daily_return_observed']=r
        # Audit raw coverage, before any fill.
        g['raw_observed']=g[['open','close','adj_close','volume']].notna().all(axis=1)
        g['coverage_63']=g.raw_observed.rolling(63,min_periods=63).mean()
        g['coverage_252']=g.raw_observed.rolling(252,min_periods=252).mean()
        g['coverage_21']=g.raw_observed.rolling(21,min_periods=21).mean()
        g['coverage_126']=g.raw_observed.rolling(126,min_periods=126).mean()
        for c in FEATURES:
            g[c]=g[c].replace([np.inf,-np.inf],np.nan)
            g[c+'_was_missing']=g[c].isna()
        # Only auxiliary ML features may be carried forward; never screening features.
        screening={'mom_2_12','vol_63','ln_adv_20'}
        for c in FEATURES:
            if c not in screening:g[c]=g[c].ffill(limit=cfg.get('max_feature_ffill',2))
        results.append(g)
    return pd.concat(results,ignore_index=True)

def decision_dates(panel,cfg):
    spy_dates=pd.DatetimeIndex(panel.loc[panel.ticker=='SPY','date'].dropna().unique()).sort_values()
    dates=spy_dates[(spy_dates>=pd.Timestamp(cfg['research_start']))&(spy_dates<=pd.Timestamp(cfg['research_end']))]
    freq={'monthly':'M','quarterly':'Q','semiannual':'2Q','annual':'Y'}[cfg['rebalance_frequency']]
    # Select last observed SPY session in each calendar period.
    # Semiannual is anchored to Jan-Jun / Jul-Dec, not a floating 2-quarter offset.
    periods = (dates.year*2 + (dates.month>6).astype(int)) if cfg['rebalance_frequency']=='semiannual' else dates.to_period(freq)
    return pd.DatetimeIndex(pd.Series(dates,index=dates).groupby(periods).last().values)

def _choose(cell,n,group):
    if group=='High':return cell.sort_values(['mom_2_12','ticker'],ascending=[False,True]).head(n)
    if group=='Low':return cell.sort_values(['mom_2_12','ticker'],ascending=[True,True]).head(n)
    med=cell.mom_2_12.median();return cell.assign(_distance=(cell.mom_2_12-med).abs()).sort_values(['_distance','ticker']).head(n).drop(columns='_distance')

def screen(panel,cfg):
    records=[];audit=[]
    for d in decision_dates(panel,cfg):
        base=panel.loc[(panel.date==d)&(~panel.ticker.isin(['SPY','QQQ']))].copy()
        # WARNING: live constituents only, not point-in-time historical members.
        base=base.loc[(base.close>=cfg['min_price'])&(base.adv_20>=cfg['min_adv20'])&(base.coverage_252>=cfg['min_coverage'])&(base.coverage_63>=cfg['min_coverage'])&(base.coverage_21>=cfg['min_coverage'])]
        base=base.loc[base[FEATURES].notna().all(axis=1)&np.isfinite(base[FEATURES]).all(axis=1)]
        base=base.sort_values(['adv_20','ticker'],ascending=[False,True]).head(cfg['liquid_universe_size'])
        info={'decision_date':d,'eligible_top300':len(base),'selected':0,'status':'insufficient'}
        if len(base)<cfg['min_eligible_stocks']:
            audit.append(info);continue
        base=base.sort_values(['mom_2_12','ticker'],ascending=[False,True]).reset_index(drop=True)
        # Exactly three approximately equal MOM buckets.
        blocks=np.array_split(np.arange(len(base)),3)
        pieces=[];ok=True
        for group,indices in zip(['High','Medium','Low'],blocks):
            mom=base.iloc[indices].sort_values(['vol_63','ticker']).reset_index(drop=True)
            for vg,subindices,n in zip(['Low','Medium','High'],np.array_split(np.arange(len(mom)),3),[3,4,3]):
                cell=mom.iloc[subindices]
                info[f'{group}_{vg}_available']=len(cell)
                if len(cell)<n:ok=False;continue
                selected=_choose(cell,n,group).copy()
                selected['momentum_group']=group;selected['volatility_group']=vg
                pieces.append(selected)
        if ok and len(pieces)==9:
            chosen=pd.concat(pieces,ignore_index=True)
            if chosen.ticker.nunique()!=30:raise AssertionError('Duplicate selection')
            records.append(chosen);info['selected']=30;info['status']='ok'
        audit.append(info)
    return (pd.concat(records,ignore_index=True) if records else pd.DataFrame()),pd.DataFrame(audit)

def attach_labels(screened,panel,cfg):
    if screened.empty:return screened
    decisions=decision_dates(panel,cfg)
    spy=panel.loc[panel.ticker=='SPY',['date']].sort_values('date')
    trading_dates=pd.DatetimeIndex(spy.date.unique())
    # Execution = next observed SPY session open, exit = next rebalance execution open.
    execution={}
    for d in decisions:
        ix=trading_dates.searchsorted(d,side='right')
        execution[d]=trading_dates[ix] if ix<len(trading_dates) else pd.NaT
    end={d:execution[decisions[i+1]] if i+1<len(decisions) else pd.NaT for i,d in enumerate(decisions)}
    # For the final research decision, determine the next true calendar-period boundary
    # from the downloaded SPY calendar (not a fixed 21/63/126-day shift).
    if len(decisions):
        last=decisions[-1]
        months={'monthly':1,'quarterly':3,'semiannual':6,'annual':12}[cfg['rebalance_frequency']]
        period_end=last.to_period('M').to_timestamp('M') + pd.DateOffset(months=months)
        period_end=pd.Timestamp(period_end).to_period('M').to_timestamp('M')
        next_period_sessions=trading_dates[(trading_dates>last)&(trading_dates<=period_end)]
        if len(next_period_sessions):
            next_close=next_period_sessions[-1]
            ix=trading_dates.searchsorted(next_close,side='right')
            if ix<len(trading_dates):end[last]=trading_dates[ix]
    out=screened.copy();out['decision_date']=out.date
    out['execution_date']=out.decision_date.map(execution);out['label_end_date']=out.decision_date.map(end)
    px=panel[['ticker','date','open','close','adj_close']].copy()
    px['adj_open']=px.open*px.adj_close/px.close.replace(0,np.nan)
    lookup=px.set_index(['ticker','date'])['adj_open']
    out['entry_adj_open']=[lookup.get((t,d),np.nan) for t,d in zip(out.ticker,out.execution_date)]
    out['exit_adj_open']=[lookup.get((t,d),np.nan) for t,d in zip(out.ticker,out.label_end_date)]
    out['target_forward_return']=out.exit_adj_open/out.entry_adj_open-1
    out['target_absolute_return']=out['target_forward_return']  # compatibility alias
    spy_lookup=px.loc[px.ticker=='SPY'].set_index('date').adj_open
    out['spy_return']=[spy_lookup.get(e,np.nan)/spy_lookup.get(s,np.nan)-1 for s,e in zip(out.execution_date,out.label_end_date)]
    qqq_lookup=px.loc[px.ticker=='QQQ'].set_index('date').adj_open
    out['qqq_return']=[qqq_lookup.get(e,np.nan)/qqq_lookup.get(s,np.nan)-1 for s,e in zip(out.execution_date,out.label_end_date)]
    out['target_spy_relative_return']=out.target_forward_return-out.spy_return
    out['target_qqq_relative_return']=out.target_forward_return-out.qqq_return
    out['split']=np.select([out.decision_date<pd.Timestamp(cfg.get('train_end_exclusive','2021-01-01')),out.decision_date<pd.Timestamp(cfg.get('validation_end_exclusive','2023-01-01'))],['train','validation'],default='test')
    out['label_status']=np.where(out.target_forward_return.notna(),'valid','missing_or_unsettled')
    return out

def run(cfg,raw=None):
    if raw is None:raw=pd.read_parquet(ROOT/'data/raw/market_data.parquet')
    if any(raw.ticker.astype(str).str.startswith('SYN')):raise ValueError('Synthetic tickers detected; real-data pipeline refuses demo input')
    panel=features(raw,cfg)
    processed=ROOT/'data/processed_with_nan';processed.mkdir(exist_ok=True,parents=True)
    panel.to_parquet(processed/'full_feature_panel.parquet',index=False)
    screened,audit=screen(panel,cfg)
    if screened.empty:raise RuntimeError('No screened dates; see audit and download coverage')
    screened=attach_labels(screened,panel,cfg)
    ready=screened.loc[screened[FEATURES+['target_forward_return','spy_return','qqq_return']].notna().all(axis=1)].copy()
    ready=ready.loc[np.isfinite(ready[FEATURES+['target_forward_return','spy_return','qqq_return']]).all(axis=1)]
    out=ROOT/'data/processed_ready'/cfg['rebalance_frequency'];out.mkdir(exist_ok=True,parents=True)
    screened.to_parquet(out/'screened_with_label_status.parquet',index=False)
    ready.to_parquet(out/'model_ready.parquet',index=False)
    (out/'feature_columns.json').write_text(json.dumps(FEATURES,indent=2))
    (out/'model_contract.json').write_text(json.dumps({'primary_target':'target_forward_return','secondary_labels':['target_spy_relative_return','target_qqq_relative_return'],'benchmarks':['SPY','QQQ'],'feature_columns':FEATURES,'split_boundaries':{'train_end_exclusive':'2021-01-01','validation_end_exclusive':'2023-01-01'},'fit_rule':'label_end_date must be <= model fit cutoff; use chronological splits'},indent=2))
    (ROOT/'outputs').mkdir(exist_ok=True)
    audit.to_csv(ROOT/'outputs'/f"screening_audit_{cfg['rebalance_frequency']}.csv",index=False)
    quality={'raw_rows':len(raw),'processed_rows':len(panel),'screened_rows':len(screened),'ready_rows':len(ready),'screened_months':int(screened.decision_date.nunique()),'ready_months':int(ready.decision_date.nunique()),'missing_label_rows':int(screened.target_absolute_return.isna().sum()),'primary_target':'target_forward_return','benchmark_tickers':['SPY','QQQ'],'universe_limitation':'CURRENT S&P 500 CONSTITUENTS ONLY: survivorship bias; NOT historical point-in-time membership'}
    (ROOT/'outputs'/f"quality_summary_{cfg['rebalance_frequency']}.json").write_text(json.dumps(quality,indent=2))
    # Never use future label availability to reselect stocks. Store original screening separately.
    assert not ready[FEATURES+['target_forward_return']].isna().any().any()
    assert not ready.duplicated(['decision_date','ticker']).any()
    return panel,screened,ready,audit
