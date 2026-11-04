"""Fetch raw data for DSA5205 Project 2 and cache it under ../data/.

Sources (lawful, public):
  - SPY daily OHLCV (split/dividend adjusted): Yahoo Finance chart API via
    urllib (yfinance used only as a fallback).
  - 3-month T-bill rate (DTB3): FRED CSV endpoint; Yahoo ^IRX as fallback.

Outputs:
  data/spy_daily.csv  -- date, open, high, low, close, volume (adjusted)
  data/dtb3.csv       -- date, dtb3 (percent, annualized)

Cache safety: both datasets are downloaded and validated BEFORE any cache file
is replaced, and each file is written to a temporary name and atomically moved
into place. A failed download therefore never destroys a valid cache, and the
backtest can always run from the committed cache without network access.

Also verifies return continuity of the adjusted series, including a per-event
check on the two most recent cash dividends: the adjusted close-to-close
return on the ex-date must equal (P_ex + D) / P_prev - 1.
"""

import io
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
START = "2004-06-01"  # a bit of history before 2005 so long lookbacks exist at split start
END = "2026-10-30"    # fixed analysis end date, so results are reproducible
SYMBOL = "SPY"
MIN_SPY_ROWS = 5000
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SPY_COLS = ["date", "open", "high", "low", "close", "volume"]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _period(start: str, end: str) -> tuple[int, int]:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    return p1, p2


def fetch_spy_yahoo_chart(start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily adjusted OHLCV from the Yahoo chart API, plus the dividend table
    used for the per-event adjustment verification."""
    p1, p2 = _period(start, END)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    raw = json.loads(_get(url))
    res = raw["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s").normalize(),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close_raw": q["close"],
            "close": adj,  # split/dividend adjusted close
            "volume": q["volume"],
        }
    )
    # scale raw OHLC by the same adjustment factor so the whole OHLC block is adjusted
    factor = df["close"] / df["close_raw"]
    for col in ["open", "high", "low"]:
        df[col] = df[col] * factor
    df = df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)

    divs = pd.DataFrame(columns=["date", "amount"])
    events = res.get("events", {}).get("dividends", {})
    if events:
        divs = pd.DataFrame(
            {
                "date": pd.to_datetime([int(k) for k in events], unit="s").normalize(),
                "amount": [v["amount"] for v in events.values()],
            }
        ).sort_values("date").reset_index(drop=True)
    return df, divs


def fetch_spy_yfinance(start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    px = yf.download(SYMBOL, start=start, end=pd.Timestamp(END) + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(px.index).normalize(),
            "open": px["Open"].to_numpy().ravel(),
            "high": px["High"].to_numpy().ravel(),
            "low": px["Low"].to_numpy().ravel(),
            "close_raw": px["Close"].to_numpy().ravel(),
            "close": px["Close"].to_numpy().ravel(),  # already adjusted; raw unavailable here
            "volume": px["Volume"].to_numpy().ravel(),
        }
    ).dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df, pd.DataFrame(columns=["date", "amount"])  # no event table in fallback


def _fred_dtb3(start: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd={start}&coed={END}"
    # curl handles the FRED endpoint more reliably than Python TLS on Windows
    out = subprocess.run(
        ["curl", "-sS", "-m", "45", "-A", UA["User-Agent"], url],
        check=True, capture_output=True,
    ).stdout
    df = pd.read_csv(io.BytesIO(out))
    df.columns = ["date", "dtb3"]
    return df


def _yahoo_irx(start: str) -> pd.DataFrame:
    """13-week T-bill yield (^IRX) from Yahoo, as a FRED DTB3 substitute."""
    p1, p2 = _period(start, END)
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EIRX"
        f"?period1={p1}&period2={p2}&interval=1d"
    )
    raw = json.loads(_get(url))
    res = raw["chart"]["result"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
            "dtb3": res["indicators"]["quote"][0]["close"],
        }
    )
    return df


def fetch_dtb3(start: str) -> tuple[pd.DataFrame, str]:
    """Annualized 3-month T-bill rate in percent. Primary: FRED DTB3;
    fallback: Yahoo ^IRX (same instrument)."""
    try:
        df, src = _fred_dtb3(start), "FRED DTB3"
    except Exception as e:  # noqa: BLE001 - FRED is occasionally unreachable
        print(f"[warn] FRED DTB3 failed ({e}); using Yahoo ^IRX instead", file=sys.stderr)
        df, src = _yahoo_irx(start), "Yahoo ^IRX"
    df["date"] = pd.to_datetime(df["date"])
    df["dtb3"] = pd.to_numeric(df["dtb3"], errors="coerce")
    df = df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df, src


def validate_spy(df: pd.DataFrame) -> None:
    """Hard validation of a freshly downloaded SPY frame; raises on any problem
    so the existing cache is never overwritten with bad data."""
    if df.empty:
        raise RuntimeError("SPY download returned no rows; cached data was not overwritten")
    missing = [c for c in SPY_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"SPY download missing columns {missing}; cache not overwritten")
    if len(df) < MIN_SPY_ROWS:
        raise RuntimeError(f"SPY download has only {len(df)} rows (< {MIN_SPY_ROWS}); cache not overwritten")
    if (df["close"] <= 0).any():
        raise RuntimeError("SPY download contains non-positive prices; cache not overwritten")
    if df["date"].duplicated().any():
        raise RuntimeError("SPY download contains duplicate dates; cache not overwritten")
    last = df["date"].iloc[-1].date()
    if str(last) != END:
        raise RuntimeError(f"SPY download ends at {last}, expected {END}; cache not overwritten")


def validate_dtb3(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("T-bill download returned no rows; cached data was not overwritten")
    if df["date"].duplicated().any():
        raise RuntimeError("T-bill download contains duplicate dates; cache not overwritten")


def adjustment_check(df: pd.DataFrame, divs: pd.DataFrame) -> None:
    """Continuity checks on the adjusted series, plus a per-event verification
    on the two most recent cash dividends: the adjusted close-to-close return
    on the ex-date must equal (P_ex + D)/P_prev - 1 (brief, Appendix B.4)."""
    ret = df["close"].pct_change().dropna()
    print(f"[check] observations: {len(df)}  range: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    print(f"[check] non-positive prices: {(df['close'] <= 0).sum()}")
    print(f"[check] duplicate dates: {df['date'].duplicated().sum()}")
    print(f"[check] |daily return| > 25% days (should be ~0 for SPY): {(ret.abs() > 0.25).sum()}")
    print(f"[check] mean |daily return|: {ret.abs().mean():.5f}  (continuous across corporate actions)")
    for _, ev in divs.tail(2).iterrows():
        idx = df.index[df["date"] == ev["date"]]
        if len(idx) == 0 or idx[0] == 0:
            continue
        i = idx[0]
        p_ex, p_prev = df.loc[i, "close_raw"], df.loc[i - 1, "close_raw"]
        manual = (p_ex + ev["amount"]) / p_prev - 1.0
        vendor = df.loc[i, "close"] / df.loc[i - 1, "close"] - 1.0
        # tolerance 1e-4: vendor prices/dividends are rounded to cents; a genuine
        # adjustment break would show up at the dividend-size scale (~3e-3)
        diff = abs(manual - vendor)
        status = "OK" if diff < 1e-4 else "MISMATCH"
        print(f"[check] dividend {ev['date'].date()} ${ev['amount']:.3f}: "
              f"manual TR {manual:.6f} vs vendor adj return {vendor:.6f} "
              f"(|diff|={diff:.2e}) -> {status}")
        if status != "OK":
            raise RuntimeError(f"adjustment verification failed on {ev['date'].date()}")


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    # 1) download everything first; on any failure the existing cache is untouched
    try:
        spy, divs = fetch_spy_yahoo_chart(START)
        src = "yahoo chart api"
    except Exception as e:  # fallback path
        print(f"[warn] yahoo chart api failed ({e}); falling back to yfinance", file=sys.stderr)
        spy, divs = fetch_spy_yfinance(START)
        src = "yfinance"
    dtb3, rf_src = fetch_dtb3(START)

    # 2) validate both datasets before replacing anything
    validate_spy(spy)
    validate_dtb3(dtb3)

    # 3) atomic cache replacement
    adjustment_check(spy, divs)
    atomic_write_csv(spy[SPY_COLS], os.path.join(DATA, "spy_daily.csv"))
    atomic_write_csv(dtb3, os.path.join(DATA, "dtb3.csv"))
    print(f"[ok] SPY daily saved ({src}): {len(spy)} rows")
    print(f"[ok] risk-free saved ({rf_src}): {len(dtb3)} rows, last date {dtb3['date'].iloc[-1].date()}")


if __name__ == "__main__":
    main()
