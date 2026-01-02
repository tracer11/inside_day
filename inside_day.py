#!/usr/bin/env python3
"""
Inside-day (intraday) scanner optimized for:
- run late day (2:45pm CT / 3:45pm ET)
- buy late day
- sell next morning

Design:
1) Signal = "today (so far) inside yesterday" with small ATR tolerance (prevents wick-noise zeroing).
2) Compression is a rank metric, NOT a hard gate (you still see strict inside names even if range expanded).
3) Dollar volume is "smart": excludes today's partial daily bar when market is open.
4) Options spread check runs only on top-N ranked candidates.
"""

import time
import random
from io import StringIO
from time import sleep
from datetime import time as dtime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from curl_cffi import requests as crequests

# ---------------- CONFIG ----------------
LOOKBACK_DAYS = 45
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
US_MARKET_TZ = "US/Eastern"  # keep everything in ET for market logic

# Intraday "still inside" tolerance:
# allow today to poke outside yesterday by ATR10 * INSIDE_TOL_ATR
INSIDE_TOL_ATR = 0.05   # 0.03 tight, 0.05 balanced, 0.08 looser

# Optional: ignore first X minutes of the day when computing today's hi/lo (to reduce opening wick noise)
IGNORE_OPEN_MINUTES = 0  # set 30 if you want to ignore 9:30-10:00 ET for today's hi/lo

# Liquidity filters
MIN_DOLLAR_VOL = 10_000_000      # lower than 25M because options spread filter is the real liquidity gate
DOLLAR_VOL_LOOKBACK = 20         # completed bars

# Options / spread filters
CHECK_OPTIONS_FOR_MAX = 150
MAX_ABS_SPREAD = 0.15
MAX_REL_SPREAD = 0.015
ATM_MONEYNES_PCT = 0.02

# Ranking weights
WEIGHT_INSIDE_TIGHTNESS = 0.55   # how tightly today stays inside yesterday (normalized by ATR)
WEIGHT_TREND = 0.25              # trend support (EMA21 + slope)
WEIGHT_SPY_REGIME = 0.10         # small regime nudge, not dictator
WEIGHT_SPREAD_BONUS = 0.20       # bonus if tight options spread found

# Trend / direction
USE_TREND_FILTER = True
SLOPE_TOL = 0.02
STRONG_SLOPE = 0.12
SPY_DIR_WEIGHT = 0.25

# Major ETFs included
MAJOR_ETFS = ["SPY", "QQQ", "IWM", "DIA"]

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------------- UTILS ----------------
def _to_float(x):
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)

def _normalize_ticker(t: str) -> str:
    return str(t).strip().replace(".", "-")

def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _market_close_time_et() -> dtime:
    # 4:00pm ET regular close (ignore half-days for simplicity)
    return dtime(16, 0)

def _market_open_time_et() -> dtime:
    return dtime(9, 30)

# ---------------- S&P 500 LIST ----------------
def get_sp500_tickers(max_retries: int = 3) -> list[str]:
    DATAHUB_CSV = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(WIKI_URL, headers=UA, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)

            candidates = []
            for df in tables:
                cols = {str(c).lower().strip() for c in df.columns}
                if ("symbol" in cols or "ticker symbol" in cols) and ("security" in cols or "company" in cols):
                    candidates.append(df)

            if not candidates:
                raise ValueError("No constituents-like table found on Wikipedia.")

            df = max(candidates, key=len)
            sym_col = "Symbol" if "Symbol" in df.columns else ("Ticker symbol" if "Ticker symbol" in df.columns else None)
            if not sym_col:
                raise ValueError("Ticker column missing in chosen table.")

            tickers = df[sym_col].astype(str).map(_normalize_ticker).dropna().unique().tolist()
            if len(tickers) < 450:
                raise ValueError(f"Too few tickers parsed: {len(tickers)}")
            return tickers

        except Exception:
            if attempt < max_retries:
                sleep(0.8)

    # fallback CSV
    resp = requests.get(DATAHUB_CSV, headers=UA, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if "Symbol" not in df.columns:
        raise ValueError("Fallback CSV missing 'Symbol' column.")
    tickers = df["Symbol"].astype(str).map(_normalize_ticker).dropna().unique().tolist()
    if len(tickers) < 450:
        raise ValueError(f"Too few tickers from fallback: {len(tickers)}")
    return tickers

# ---------------- TECHNICALS ----------------
def atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    parts = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    tr = parts.max(axis=1, skipna=True)
    return tr.rolling(period).mean()

def ema(series: pd.Series, length: int = 21) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

# ---------------- DAILY DOWNLOAD ----------------
def _split_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _clean_ohlcv(sub: pd.DataFrame) -> pd.DataFrame | None:
    if sub is None or sub.empty:
        return None
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(set(sub.columns)):
        return None
    sub = sub[needed].dropna()
    return None if sub.empty else sub

def download_daily_spx(tickers):
    sess = crequests.Session(impersonate="chrome")

    data = {}
    failed = []
    BATCH_SIZE = 50
    MAX_RETRIES = 4
    TIMEOUT_S = 30

    for batch_idx, batch in enumerate(_split_chunks(tickers, BATCH_SIZE), 1):
        remaining = batch[:]
        attempt = 0

        while remaining and attempt <= MAX_RETRIES:
            try:
                df = yf.download(
                    " ".join(remaining),
                    period=f"{LOOKBACK_DAYS}d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    session=sess,
                    timeout=TIMEOUT_S,
                    threads=True,
                    group_by="ticker",
                )
                if df is None or df.empty:
                    raise RuntimeError("empty batch df")

                still_missing = []
                for t in remaining:
                    try:
                        if isinstance(df.columns, pd.MultiIndex) and t in df.columns.get_level_values(0):
                            sub = df[t].copy()
                        else:
                            sub = df.copy()

                        sub = _clean_ohlcv(sub)
                        if sub is None:
                            still_missing.append(t)
                            continue

                        data[t] = sub
                    except Exception:
                        still_missing.append(t)

                remaining = still_missing
                attempt += 1

                if remaining:
                    sleep_s = (0.8 * (2 ** (attempt - 1))) + random.random() * 0.8
                    time.sleep(sleep_s)

            except Exception:
                attempt += 1
                sleep_s = (0.8 * (2 ** (attempt - 1))) + random.random() * 0.8
                time.sleep(sleep_s)

        failed.extend(remaining)
        done = min(batch_idx * BATCH_SIZE, len(tickers))
        print(f"...batch {batch_idx}: downloaded {len(data)} / {done}")
        time.sleep(0.4 + random.random() * 0.5)

    if failed:
        print(f"\nFailed downloads ({len(set(failed))}): {sorted(set(failed))}")

    return data

# ---------------- INTRADAY OVERRIDE ----------------
def override_today_with_intraday(ticker: str, df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    If last daily bar is today, overwrite today's (daily) High/Low/Close with intraday extremes/last.
    Optionally ignore first N minutes to avoid opening wick noise.
    """
    if df_daily is None or df_daily.empty:
        return df_daily

    now_et = pd.Timestamp.now(tz=US_MARKET_TZ)
    today_date = now_et.date()

    last_ts = df_daily.index[-1]
    last_date = last_ts.date()
    if last_date != today_date:
        return df_daily

    try:
        intraday = yf.download(
            ticker,
            period="2d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            timeout=30,
            threads=False,
        )
    except Exception:
        return df_daily

    if intraday is None or intraday.empty:
        return df_daily

    if intraday.index.tz is None:
        intraday = intraday.tz_localize("UTC").tz_convert(US_MARKET_TZ)
    else:
        intraday = intraday.tz_convert(US_MARKET_TZ)

    intraday_today = intraday[intraday.index.date == today_date]
    if intraday_today.empty:
        return df_daily

    if IGNORE_OPEN_MINUTES and IGNORE_OPEN_MINUTES > 0:
        start = pd.Timestamp.combine(
            pd.Timestamp(today_date).tz_localize(US_MARKET_TZ).date(),
            _market_open_time_et()
        ).tz_localize(US_MARKET_TZ) + pd.Timedelta(minutes=IGNORE_OPEN_MINUTES)
        intraday_today = intraday_today[intraday_today.index >= start]
        if intraday_today.empty:
            return df_daily

    hi = _to_float(intraday_today["High"].max())
    lo = _to_float(intraday_today["Low"].min())
    close = _to_float(intraday_today["Close"].iloc[-1])

    df_daily.iloc[-1, df_daily.columns.get_loc("High")] = hi
    df_daily.iloc[-1, df_daily.columns.get_loc("Low")] = lo
    df_daily.iloc[-1, df_daily.columns.get_loc("Close")] = close
    return df_daily

# ---------------- OPTION SPREAD CHECK ----------------
def best_option_spread_for_ticker(ticker: str, spot: float):
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None
        nearest = expirations[0]
        chain = tk.option_chain(nearest)
    except Exception:
        return None

    best_spread = None

    def scan_df(df):
        nonlocal best_spread
        if df is None or df.empty:
            return
        for _, row in df.iterrows():
            strike = row.get("strike")
            bid = row.get("bid")
            ask = row.get("ask")
            if strike is None or bid is None or ask is None:
                continue
            if pd.isna(bid) or pd.isna(ask):
                continue
            if spot == 0:
                continue
            if abs(float(strike) - float(spot)) / float(spot) > ATM_MONEYNES_PCT:
                continue

            spread = float(ask) - float(bid)
            if spread <= 0:
                continue

            mid = (float(ask) + float(bid)) / 2.0
            abs_ok = spread <= MAX_ABS_SPREAD
            rel_ok = mid > 0 and (spread / mid) <= MAX_REL_SPREAD

            if abs_ok or rel_ok:
                if best_spread is None or spread < best_spread:
                    best_spread = spread

    scan_df(chain.calls if hasattr(chain, "calls") else None)
    scan_df(chain.puts if hasattr(chain, "puts") else None)
    return best_spread

# ---------------- MARKET CONTEXT (SPY) ----------------
def get_spy_context():
    spy = yf.download("SPY", period="120d", interval="1d", progress=False, auto_adjust=False, timeout=30)
    if spy is None or spy.empty:
        return None
    spy["EMA21"] = ema(spy["Close"], 21)

    close_val = _to_float(spy["Close"].iloc[-1])
    ema_val = _to_float(spy["EMA21"].iloc[-1])

    if len(spy) > 5:
        slope_raw = float(spy["EMA21"].iloc[-1] - spy["EMA21"].iloc[-6])
    else:
        slope_raw = float(spy["EMA21"].iloc[-1] - spy["EMA21"].iloc[0])

    return {
        "close": close_val,
        "above_ema21": bool(close_val > ema_val),
        "ema21_slope_raw": slope_raw,
    }

# ---------------- INSIDE (INTRADAY) ----------------
def inside_with_tolerance(df: pd.DataFrame, idx: int, tol_atr: float) -> tuple[bool, float]:
    """
    Returns (is_inside-ish, inside_tightness_score)
    - inside-ish: today within yesterday +/- tol
    - tightness score: higher = more tightly inside (normalized by ATR)
    """
    prev_high = float(df["High"].iloc[idx - 1])
    prev_low = float(df["Low"].iloc[idx - 1])
    hi = float(df["High"].iloc[idx])
    lo = float(df["Low"].iloc[idx])

    atrv = df["ATR10"].iloc[idx]
    if pd.isna(atrv) or float(atrv) <= 0:
        return (False, 0.0)

    tol = float(atrv) * float(tol_atr)

    inside = (hi <= prev_high + tol) and (lo >= prev_low - tol)

    # Tightness: how far the bar is from breaking, normalized by ATR
    # margin_high = (prev_high - hi)  (positive if hi below prev_high)
    # margin_low  = (lo - prev_low)   (positive if lo above prev_low)
    margin_high = (prev_high - hi)
    margin_low = (lo - prev_low)

    # allow tolerance; margins can be slightly negative but still "inside-ish"
    # Convert to a bounded [0..1]-ish score
    tight = (margin_high + margin_low) / float(atrv)
    tight = max(-1.0, min(1.0, tight))
    tight_score = (tight + 1.0) / 2.0  # map [-1..1] -> [0..1]

    return (inside, float(tight_score))

# ---------------- MAIN SCAN ----------------
def scan_inside_days():
    spy_ctx = get_spy_context()

    spx = get_sp500_tickers()
    for etf in MAJOR_ETFS:
        if etf not in spx:
            spx.append(etf)

    print(f"Got {len(spx)} tickers (S&P 500 + major ETFs). Downloading daily bars...")
    ohlc_map = download_daily_spx(spx)
    print(f"Downloaded OK for {len(ohlc_map)} tickers.")

    candidates = []

    # diagnostics
    inside_pass = 0
    inside_fail = 0
    killed_by_dv = 0

    for t, df in ohlc_map.items():
        if df is None or df.empty or len(df) < 30:
            continue

        df = override_today_with_intraday(t, df)

        # indicators
        df["ATR10"] = atr(df, 10)
        df["EMA21"] = ema(df["Close"], 21)

        last_idx = len(df) - 1
        if last_idx < 1:
            continue

        # Must have yesterday too
        today = df.iloc[last_idx]
        yesterday = df.iloc[last_idx - 1]

        # inside check (with tolerance)
        inside_ok, inside_tightness = inside_with_tolerance(df, last_idx, INSIDE_TOL_ATR)
        if not inside_ok:
            inside_fail += 1
            continue
        inside_pass += 1

        atr10_val = df["ATR10"].iloc[last_idx]
        if pd.isna(atr10_val) or float(atr10_val) <= 0:
            continue

        today_range = float(today["High"] - today["Low"])
        compression_ratio = today_range / float(atr10_val)  # smaller is tighter
        compression_score = max(0.0, 1.0 - compression_ratio)  # keep for ranking only

        # ---- SMART $VOLUME FILTER (exclude today's partial bar when market is open) ----
        try:
            now_et = pd.Timestamp.now(tz=US_MARKET_TZ)
            today_date = now_et.date()
            last_date = df.index[-1].date()

            market_still_open = now_et.time() < _market_close_time_et()  # 4:00pm ET
            exclude_last = (last_date == today_date) and market_still_open

            if exclude_last:
                close_slice = df["Close"].iloc[-(DOLLAR_VOL_LOOKBACK + 1):-1]
                vol_slice = df["Volume"].iloc[-(DOLLAR_VOL_LOOKBACK + 1):-1]
            else:
                close_slice = df["Close"].iloc[-DOLLAR_VOL_LOOKBACK:]
                vol_slice = df["Volume"].iloc[-DOLLAR_VOL_LOOKBACK:]

            dv = (close_slice * vol_slice).dropna()
            avg_dv = float(dv.mean()) if not dv.empty else 0.0
        except Exception:
            avg_dv = 0.0

        if avg_dv < MIN_DOLLAR_VOL and t not in MAJOR_ETFS:
            killed_by_dv += 1
            continue
        # ---------------------------------------------------------------------

        # trend features (rank + direction)
        above_ema = bool(today["Close"] > df["EMA21"].iloc[last_idx]) if USE_TREND_FILTER else True

        ema21_series = df["EMA21"]
        if len(ema21_series) > 5:
            ema21_slope_raw = float(ema21_series.iloc[last_idx] - ema21_series.iloc[last_idx - 5])
        else:
            ema21_slope_raw = float(ema21_series.iloc[last_idx] - ema21_series.iloc[0])

        ema21_slope_norm = float(ema21_slope_raw) / float(atr10_val)

        # SPY regime (small nudge)
        market_ok = True
        spy_bias = 0.0
        if spy_ctx is not None:
            market_ok = bool(spy_ctx["above_ema21"])
            spy_bias = 1.0 if market_ok else -1.0

        candidates.append(
            {
                "ticker": t,
                "date": df.index[last_idx].date().isoformat(),
                "prev_high": round(float(yesterday["High"]), 4),
                "prev_low": round(float(yesterday["Low"]), 4),
                "close": round(float(today["Close"]), 4),
                "atr10": round(float(atr10_val), 4),
                "today_range": round(float(today_range), 4),
                "compression_score": round(float(compression_score), 4),
                "inside_tightness": round(float(inside_tightness), 4),
                "above_ema": above_ema,
                "market_ok": market_ok,
                "ema21_slope_norm": round(float(ema21_slope_norm), 4),
                "ema21_slope_norm_raw": float(ema21_slope_norm),
                "avg_dollar_vol": round(float(avg_dv), 0),
                "spy_bias": spy_bias,
            }
        )

    print(f"Found {len(candidates)} inside-day candidates.")
    print(f"diagnostics: inside_pass={inside_pass} inside_fail={inside_fail} killed_by_dv={killed_by_dv}")

    # ---- SCORE + DIRECTION (rank first, then options spread) ----
    scored = []
    for sig in candidates:
        # direction logic
        stock_bias = 1.0 if sig["above_ema"] else -1.0
        slope = float(sig["ema21_slope_norm_raw"])
        slope_bias = _clamp(slope, -1.0, 1.0)
        combined = 0.6 * stock_bias + 0.4 * slope_bias

        # SPY tiebreaker only when stock trend isn't strong
        if spy_ctx is not None and abs(slope) < STRONG_SLOPE:
            combined = (1.0 - SPY_DIR_WEIGHT) * combined + SPY_DIR_WEIGHT * float(sig["spy_bias"])

        direction = "CALL" if combined >= 0 else "PUT"

        # HARD GATE: don't fight EMA drift
        if direction == "CALL" and slope < -SLOPE_TOL:
            continue
        if direction == "PUT" and slope > SLOPE_TOL:
            continue

        # rank score
        trend_score = 1.0 if sig["above_ema"] else 0.0
        slope_score = (slope_bias + 1.0) / 2.0  # map [-1..1] to [0..1]

        spy_score = 1.0 if sig["market_ok"] else 0.0

        base = (
            WEIGHT_INSIDE_TIGHTNESS * float(sig["inside_tightness"])
            + WEIGHT_TREND * (0.5 * trend_score + 0.5 * slope_score)
            + WEIGHT_SPY_REGIME * spy_score
            + 0.10 * float(sig["compression_score"])  # small extra bias; not a gate
        )

        sig2 = sig.copy()
        sig2["direction"] = direction
        sig2["score"] = round(float(base), 4)
        scored.append(sig2)

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    # ---- OPTIONS SPREAD CHECK ----
    tight = []
    for i, sig in enumerate(scored[:CHECK_OPTIONS_FOR_MAX]):
        best_spread = best_option_spread_for_ticker(sig["ticker"], float(sig["close"]))
        if best_spread is None:
            continue

        sig2 = sig.copy()
        sig2["best_spread"] = round(float(best_spread), 4)

        # bonus if tight spread
        sig2["score"] = round(float(sig2["score"] + WEIGHT_SPREAD_BONUS), 4)
        tight.append(sig2)

    tight = sorted(tight, key=lambda x: x["score"], reverse=True)
    return scored, tight, spy_ctx

if __name__ == "__main__":
    all_setups, tight_setups, spy_ctx = scan_inside_days()

    # Tier A cutoff
    TIER_A_SCORE = 0.55
    tier_a = [r for r in tight_setups if r["score"] >= TIER_A_SCORE]

    if spy_ctx is not None:
        print(f"\nSPY context: close={spy_ctx['close']:.2f} above_ema21={spy_ctx['above_ema21']}")

    if tier_a:
        print("\nTIER A (score >= 0.55): inside-day setups WITH tight options (late-day / next-morning)")
        cols = "ticker | score | dir  | date       | close   | ATR10 | spread | inside_tight | comp | slope_norm | $vol(M)"
        print(cols)
        for r in tier_a:
            dollar_vol_m = (float(r.get("avg_dollar_vol", 0.0)) / 1_000_000.0)
            print(
                f"{r['ticker']:5s} | {r['score']:>5} | {r['direction']:<4} | {r['date']} | "
                f"{r['close']:>7} | {r['atr10']:>5} | {r.get('best_spread','NA'):>6} | "
                f"{r.get('inside_tightness','NA'):>11} | {r.get('compression_score','NA'):>4} | "
                f"{r.get('ema21_slope_norm','NA'):>9} | {dollar_vol_m:>6.1f}"
            )
    else:
        print("\nTIER A (score >= 0.55): none")
