"""Feature and label construction for the SPY daily timing strategy.

Timing convention (leakage control, executable trading times):
  - The decision is made AFTER the close of day t, using only information
    available up to and including that close.
  - The order is executed at the OPEN of day t+1 (the first executable price
    after the signal is known -- we never trade at the same close whose data
    generated the signal).
  - The position is held from open t+1 to open t+2, so the label is the
    open-to-open return  y_t = Open_{t+2}/Open_{t+1} - 1.

Return accounting: the models predict the EXCESS label  y_t - r_f,t  and the
portfolio return is  r_p = r_f + pi_t * (y_t - r_f,t) - cost * |pi_t - pi_{t-1}|,
i.e. capital not invested in SPY earns the T-bill rate, leveraged positions
borrow at r_f, and short proceeds are invested at r_f (SPY borrow fees are
negligible for an easy-to-borrow ETF; stated as an assumption in the report).

Features (all computed from adjusted OHLCV known at close t):
  ret_1        1-day close-to-close return r_t
  mom_5        5-day cumulative return
  mom_10       10-day cumulative return
  mom_21       21-day cumulative return
  vol_5        5-day realized volatility (std of daily returns)
  vol_21       21-day realized volatility
  ma_ratio_10  Close_t / MA10_t - 1
  ma_ratio_50  Close_t / MA50_t - 1
  ma_ratio_200 Close_t / MA200_t - 1
  rsi_14       14-day relative strength index (0-100, centered later by standardization)
  range_hl     (High_t - Low_t) / Close_t
  vol_chg      log(Volume_t) - mean of log(Volume) over the past 21 days
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

FEATURES = [
    "ret_1", "mom_5", "mom_10", "mom_21",
    "vol_5", "vol_21",
    "ma_ratio_10", "ma_ratio_50", "ma_ratio_200",
    "rsi_14", "range_hl", "vol_chg",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def load_spy() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, "spy_daily.csv"), parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_rf() -> pd.Series:
    """Daily risk-free rate (simple), from the annualized 3-month T-bill percent."""
    rf = pd.read_csv(os.path.join(DATA, "dtb3.csv"), parse_dates=["date"])
    rf = rf.sort_values("date").set_index("date")["dtb3"]
    return rf / 100.0 / 252.0


def build_dataset() -> pd.DataFrame:
    """Return a DataFrame indexed by decision date t (close of day t) with the
    FEATURES, the label `fwd_ret` (open t+1 -> open t+2 return), the matching
    risk-free accrual `rf_next`, the excess label `fwd_excess`, the 21-day
    realized close-to-close variance `sigma2_21` (known at close t, used for
    position sizing), `label_end_date` (the date on which the label is fully
    realized -- the open of day t+2 -- used to exclude unrealized labels from
    every fit and selection sample), and `naive_excess` (the recursive
    historical-mean forecast using only labels realized by close t)."""
    df = load_spy()
    open_, close = df["open"], df["close"]
    high, low, vol = df["high"], df["low"], df["volume"]
    ret = close.pct_change()

    out = pd.DataFrame({"date": df["date"]})
    out["ret_1"] = ret
    for w in (5, 10, 21):
        out[f"mom_{w}"] = close / close.shift(w) - 1.0
    out["vol_5"] = ret.rolling(5).std()
    out["vol_21"] = ret.rolling(21).std()
    for w in (10, 50, 200):
        out[f"ma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0
    out["rsi_14"] = _rsi(close, 14)
    out["range_hl"] = (high - low) / close
    logv = np.log(vol.replace(0, np.nan))
    out["vol_chg"] = logv - logv.rolling(21).mean()

    out["sigma2_21"] = ret.rolling(21).var()               # known at close t
    # label: position decided at close t is filled at open t+1 and held to open t+2
    out["fwd_ret"] = open_.shift(-2) / open_.shift(-1) - 1.0
    # the label of decision day t is only fully realized at the open of day t+2;
    # store that date so fitting/selection can exclude unrealized labels
    out["label_end_date"] = df["date"].shift(-2)
    out = out.dropna(subset=FEATURES + ["sigma2_21", "fwd_ret"]).reset_index(drop=True)

    rf_daily = load_rf()
    # the T-bill yield quoted at close t is known at decision time and accrues
    # over the holding period (open t+1 -> open t+2)
    out = out.merge(rf_daily.rename("rf_next").reset_index(), on="date", how="left")
    out["rf_next"] = out["rf_next"].ffill().fillna(0.0)
    out["fwd_excess"] = out["fwd_ret"] - out["rf_next"]
    # recursive (expanding) historical mean of the excess label, using only
    # labels fully realized by close t (y_s is realized at open s+2, so only
    # s <= t-2 qualifies) -- the Campbell-Thompson style naive forecast.
    out["naive_excess"] = out["fwd_excess"].expanding().mean().shift(2)
    return out


if __name__ == "__main__":
    ds = build_dataset()
    print(ds.describe().T.round(5))
    print(f"rows: {len(ds)}  range: {ds['date'].iloc[0].date()} -> {ds['date'].iloc[-1].date()}")
