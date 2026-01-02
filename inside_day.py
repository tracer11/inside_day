#!/usr/bin/env python3
import time
import random
from io import StringIO
from time import sleep

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from curl_cffi import requests as crequests  # required by your yfinance build

# ---------------- CONFIG ----------------
LOOKBACK_DAYS = 35
RANGE_COMPRESSION_PCT = 0.70
USE_TREND_FILTER = True
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
US_MARKET_TZ = "US/Eastern"

# options / liquidity filters
CHECK_OPTIONS_FOR_MAX = 150
MAX_ABS_SPREAD = 0.15
MAX_REL_SPREAD = 0.015
ATM_MONEYNES_PCT = 0.02

# scoring weights (keep simple)
WEIGHT_COMPRESSION = 0.5
WEIGHT_TREND = 0.3
WEIGHT_SPREAD = 0.2

# Major ETFs explicitly included in the scan
MAJOR_ETFS = ["SPY", "QQQ", "IWM", "DIA"]

# ---- WIN-RATE OPTIMIZERS ----
# 1) Hard gate: don't take CALLs when EMA drift is meaningfully negative (and vice versa)
SLOPE_TOL = 0.02          # ATR-normalized tolerance (noise band)

# 2) SPY as tiebreaker only (not dictator)
STRONG_SLOPE = 0.12       # if |slope_norm| >= this, ignore SPY for direction
SPY_DIR_WEIGHT = 0.25     # if slope is weak, SPY nudges direction by this amount

# 3) Correlation control (prevents stacked red days)
ENABLE_SECTOR_CAP = True
MAX_PER_SECTOR = 1        # 1 = max win-rate / least correlation; 2 if you want more trades
SECTOR_LOOKUP_MAX = 60    # only lookup sectors for top-N candidates (keeps runtime sane)

# 4) Optional: avoid dead names (helps follow-through)
MIN_DOLLAR_VOL = 25_000_000  # avg(Volume)*Close over last ~20 bars
DOLLAR_VOL_LOOKBACK = 20

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# -------------- UTILS --------------
def _to_float(x):
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)

def _normalize_ticker(t: str) -> str:
    return str(t).strip().replace(".", "-")

def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _safe_mean(series: pd.Series) -> float:
    try:
        v = float(series.dropna().mean())
        return v
    except Exception:
        return float("nan")

# -------------- S&P 500 LIST (robust) --------------
def get_sp500_tickers(max_retries: int = 3) -> list[str]:
    DATAHUB_CSV = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(WIKI_URL, headers=UA, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)

            candidates = []
            for df in tables:
                cols = {str(c).lower().strip() for c in df.columns}
                if ("symbol" in cols or "ticker symbol" in cols) and (
                    "security" in cols or "company" in cols
                ):
                    candidates.append(df)

            if not candidates:
                raise ValueError("No constituents-like table found on Wikipedia.")

            df = max(candidates, key=len)
            sym_col = (
                "Symbol"
                if "Symbol" in df.columns
                else ("Ticker symbol" if "Ticker symbol" in df.columns else None)
            )
            if not sym_col:
                raise ValueError("Ticker column missing in chosen table.")

            tickers = df[sym_col].astype(str).map(_normalize_ticker).dropna().unique().tolist()
            if len(tickers) < 450:
                raise ValueError(f"Too few tickers parsed from Wikipedia: {len(tickers)}")
            return tickers

        except Exception as e:
            last_err = e
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

# -------------- TECHNICALS --------------
def atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    parts = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    tr = parts.max(axis=1, skipna=True)
    return tr.rolling(period).mean()

def ema(series: pd.Series, length: int = 21) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def is_inside_day(df: pd.DataFrame, idx: int) -> bool:
    return (
        df["High"].iloc[idx] <= df["High"].iloc[idx - 1]
        and df["Low"].iloc[idx] >= df["Low"].iloc[idx - 1]
    )

# -------------- DAILY DOWNLOAD (batch + curl_cffi session) --------------
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
        time.sleep(0.6 + random.random() * 0.6)

    if failed:
        print(f"\nFailed downloads ({len(set(failed))}): {sorted(set(failed))}")

    return data

# -------------- INTRADAY OVERRIDE --------------
def override_today_with_intraday(ticker: str, df_daily: pd.DataFrame) -> pd.DataFrame:
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

    hi = _to_float(intraday_today["High"].max())
    lo = _to_float(intraday_today["Low"].min())
    close = _to_float(intraday_today["Close"].iloc[-1])

    df_daily.iloc[-1, df_daily.columns.get_loc("High")] = hi
    df_daily.iloc[-1, df_daily.columns.get_loc("Low")] = lo
    df_daily.iloc[-1, df_daily.columns.get_loc("Close")] = close

    return df_daily

# -------------- OPTION SPREAD CHECK --------------
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
            if abs(strike - spot) / spot > ATM_MONEYNES_PCT:
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

# -------------- MARKET CONTEXT (SPY) --------------
def get_spy_context():
    spy = yf.download("SPY", period="90d", interval="1d", progress=False, auto_adjust=False, timeout=30)
    if spy is None or spy.empty:
        return None
    spy["EMA21"] = ema(spy["Close"], 21)
    close_val = _to_float(spy["Close"].iloc[-1])
    ema_val = _to_float(spy["EMA21"].iloc[-1])

    # also compute slope to avoid "barely above EMA" regimes
    if len(spy) > 5:
        slope_raw = float(spy["EMA21"].iloc[-1] - spy["EMA21"].iloc[-6])
    else:
        slope_raw = float(spy["EMA21"].iloc[-1] - spy["EMA21"].iloc[0])

    return {
        "close": close_val,
        "above_ema21": bool(close_val > ema_val),
        "ema21_slope_raw": slope_raw,
    }

# -------------- OPTIONAL: SECTOR LOOKUP (for correlation control) --------------
_sector_cache = {}
def get_sector_fast(ticker: str) -> str:
    if ticker in _sector_cache:
        return _sector_cache[ticker]
    try:
        info = yf.Ticker(ticker).fast_info
        # fast_info doesn't include sector; fall back to .info (slower)
        sector = yf.Ticker(ticker).info.get("sector", "Unknown")
    except Exception:
        sector = "Unknown"
    _sector_cache[ticker] = sector or "Unknown"
    return _sector_cache[ticker]

# -------------- MAIN SCAN --------------
def scan_inside_days():
    spy_ctx = get_spy_context()

    spx = get_sp500_tickers()
    for etf in MAJOR_ETFS:
        if etf not in spx:
            spx.append(etf)

    print(f"Got {len(spx)} tickers (S&P 500 + major ETFs). Downloading daily bars...")
    ohlc_map = download_daily_spx(spx)
    print(f"Downloaded OK for {len(ohlc_map)} tickers.")

    all_inside_days = []

    for t, df in ohlc_map.items():
        if len(df) < 22:
            continue

        df = override_today_with_intraday(t, df)

        df["ATR10"] = atr(df, 10)
        df["EMA21"] = ema(df["Close"], 21)

        last_idx = len(df) - 1
        if last_idx < 1:
            continue

        if not is_inside_day(df, last_idx):
            continue

        today = df.iloc[last_idx]
        yesterday = df.iloc[last_idx - 1]

        atr10_val = df["ATR10"].iloc[last_idx]
        if np.isnan(atr10_val) or atr10_val == 0:
            continue

        today_range = float(today["High"] - today["Low"])
        if today_range > RANGE_COMPRESSION_PCT * float(atr10_val):
            continue

        # Dollar volume filter (helps follow-through)
        try:
            dv = (df["Close"].iloc[-DOLLAR_VOL_LOOKBACK:] * df["Volume"].iloc[-DOLLAR_VOL_LOOKBACK:]).dropna()
            avg_dv = float(dv.mean()) if not dv.empty else 0.0
        except Exception:
            avg_dv = 0.0
        if avg_dv < MIN_DOLLAR_VOL and t not in MAJOR_ETFS:
            continue

        above_ema = bool(today["Close"] > df["EMA21"].iloc[last_idx]) if USE_TREND_FILTER else True

        # EMA21 slope over ~5 sessions, normalized by ATR
        ema21_series = df["EMA21"]
        if len(ema21_series) > 5:
            ema21_slope_raw = float(ema21_series.iloc[last_idx] - ema21_series.iloc[last_idx - 5])
        else:
            ema21_slope_raw = float(ema21_series.iloc[last_idx] - ema21_series.iloc[0])

        try:
            ema21_slope_norm = float(ema21_slope_raw) / float(atr10_val)
        except Exception:
            ema21_slope_norm = 0.0

        compression_ratio = today_range / float(atr10_val)
        compression_score = max(0.0, 1.0 - compression_ratio)

        # Market regime penalty (context only)
        market_ok = True
        if spy_ctx is not None:
            # require SPY above EMA21 for "friendly" regime; otherwise penalize score
            if not spy_ctx["above_ema21"]:
                market_ok = False

        all_inside_days.append(
            {
                "ticker": t,
                "date": df.index[last_idx].date().isoformat(),
                "prev_high": round(float(yesterday["High"]), 4),
                "prev_low": round(float(yesterday["Low"]), 4),
                "close": round(float(today["Close"]), 4),
                "atr10": round(float(atr10_val), 4),
                "today_range": round(float(today_range), 4),
                "compression_score": round(float(compression_score), 4),
                "above_ema": above_ema,
                "market_ok": market_ok,
                "ema21_slope": round(float(ema21_slope_raw), 6),
                "ema21_slope_norm": round(float(ema21_slope_norm), 4),
                "ema21_slope_norm_raw": float(ema21_slope_norm),
                "avg_dollar_vol": round(float(avg_dv), 0),
            }
        )

    print(f"Found {len(all_inside_days)} inside-day candidates.")

    # ---- SCORE + DIRECTION (best hybrid: stock-first + SPY tiebreaker) ----
    filtered = []
    spy_bias = 0.0
    if spy_ctx is not None:
        spy_bias = 1.0 if spy_ctx["above_ema21"] else -1.0

    for sig in all_inside_days:
        base = WEIGHT_COMPRESSION * float(sig["compression_score"])
        if sig["above_ema"]:
            base += WEIGHT_TREND * 1.0
        if not sig["market_ok"]:
            base *= 0.70  # regime penalty only

        stock_bias = 1.0 if sig["above_ema"] else -1.0
        slope = float(sig.get("ema21_slope_norm_raw", sig["ema21_slope_norm"]))
        slope_bias = _clamp(slope, -1.0, 1.0)

        combined = 0.6 * stock_bias + 0.4 * slope_bias

        # SPY tiebreaker only when stock trend isn't strong
        if spy_ctx is not None and abs(slope) < STRONG_SLOPE:
            combined = (1.0 - SPY_DIR_WEIGHT) * combined + SPY_DIR_WEIGHT * spy_bias

        direction = "CALL" if combined >= 0 else "PUT"

        # HARD GATE: don't fight EMA drift
        if direction == "CALL" and slope < -SLOPE_TOL:
            continue
        if direction == "PUT" and slope > SLOPE_TOL:
            continue

        sig["direction"] = direction
        sig["score"] = round(float(base), 4)
        filtered.append(sig)

    all_inside_days = filtered
    by_ticker = {sig["ticker"]: sig for sig in all_inside_days}

    # ---- CHECK SPREADS AND BUILD TIER A ----
    tight = []
    all_sorted = sorted(all_inside_days, key=lambda x: x["score"], reverse=True)

    for i, sig in enumerate(all_sorted):
        if i >= CHECK_OPTIONS_FOR_MAX:
            break
        best_spread = best_option_spread_for_ticker(sig["ticker"], sig["close"])
        if best_spread is None:
            continue

        base_sig = by_ticker[sig["ticker"]].copy()
        base_sig["best_spread"] = round(float(best_spread), 4)
        base_sig["score"] = round(float(base_sig["score"] + WEIGHT_SPREAD), 4)
        tight.append(base_sig)

    tight = sorted(tight, key=lambda x: x["score"], reverse=True)

    # ---- OPTIONAL: SECTOR CAP FOR CORRELATION CONTROL ----
    if ENABLE_SECTOR_CAP and tight:
        # lookup sectors for only top N to avoid slowing everything down
        topN = tight[:SECTOR_LOOKUP_MAX]
        for r in topN:
            r["sector"] = get_sector_fast(r["ticker"])
        for r in tight[SECTOR_LOOKUP_MAX:]:
            r["sector"] = "Unknown"

        capped = []
        counts = {}
        for r in tight:
            sec = r.get("sector", "Unknown")
            counts.setdefault(sec, 0)
            if counts[sec] >= MAX_PER_SECTOR:
                continue
            counts[sec] += 1
            capped.append(r)
        tight = capped

    all_inside_days = sorted(all_inside_days, key=lambda x: x["score"], reverse=True)
    return all_inside_days, tight, spy_ctx

if __name__ == "__main__":
    all_setups, tight_setups, spy_ctx = scan_inside_days()

    tier_a = [r for r in tight_setups if r["score"] >= 0.55]

    if spy_ctx is not None:
        print(f"\nSPY context: close={spy_ctx['close']:.2f} above_ema21={spy_ctx['above_ema21']}")

    if tier_a:
        print("\nTIER A (score >= 0.55): inside-day setups WITH tight options (best hybrid)")
        cols = "ticker | score | dir  | date       | close   | ATR10 | spread | slope_norm | $vol(M) | sector"
        print(cols)
        for r in tier_a:
            dollar_vol_m = (float(r.get("avg_dollar_vol", 0.0)) / 1_000_000.0)
            sector = r.get("sector", "NA")
            print(
                f"{r['ticker']:5s} | {r['score']:>5} | {r['direction']:<4} | {r['date']} | "
                f"{r['close']:>7} | {r['atr10']:>5} | {r.get('best_spread','NA'):>6} | "
                f"{r.get('ema21_slope_norm','NA'):>9} | {dollar_vol_m:>6.1f} | {sector}"
            )
    else:
        print("\nTIER A (score >= 0.55): none")
