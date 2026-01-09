#!/usr/bin/env python3
import time
import random
from io import StringIO
from time import sleep

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from curl_cffi import requests as crequests  # Option B: required by your yfinance build

# ---------------- CONFIG ----------------
LOOKBACK_DAYS = 30
RANGE_COMPRESSION_PCT = 0.8  # run wider net, then tag which would pass 0.7
CORE_COMPRESSION_PCT = 0.7   # "core07" tag threshold

USE_TREND_FILTER = True
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
US_MARKET_TZ = "US/Eastern"

# option-spread filters
CHECK_OPTIONS_FOR_MAX = 120        # how many inside-day tickers to actually hit options API for
MAX_ABS_SPREAD = 0.15              # e.g. $0.15 max
MAX_REL_SPREAD = 0.015             # 1.5% of mid
ATM_MONEYNES_PCT = 0.02            # scan strikes within ±2% of stock price

# scoring weights
WEIGHT_COMPRESSION = 0.5
WEIGHT_TREND = 0.3
WEIGHT_SPREAD = 0.2

# NEW: reward stronger (cleaner) trends by slope magnitude (0..1)
WEIGHT_SLOPE_STRENGTH = 0.12

# Major ETFs explicitly included in the scan
MAJOR_ETFS = ["SPY", "QQQ", "IWM", "DIA"]

# ---- EMA slope hard-gate tolerance (normalized by ATR10) ----
SLOPE_TOL = 0.02

# ---------------- UPGRADE KNOBS (your 3 changes) ----------------
# 1) SPY slope regime gating (BULL/BEAR/NEUTRAL)
SPY_EMA_LEN = 21
SPY_SLOPE_LOOKBACK = 5
SPY_SLOPE_EPS = 0.00015   # normalized slope threshold (~0.015% over lookback). tune 0.0001-0.0003

# 1b) ticker EMA alignment gating
REQUIRE_TICKER_ALIGN = True

# 2) liquidity + movement sanity
MIN_DOLLAR_VOL_20 = 300_000_000  # avg(close*vol) over 20d
MIN_ATR_PCT = 0.015              # ATR10/close >= 1.5%

# 3) not-extended vs inside-day midpoint
MAX_DIST_TO_RANGE_ATR = 0.60     # abs(close - midpoint(prev_high, prev_low)) / ATR10 <= 0.60

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# -------------- UTILS --------------
def _to_float(x):
    # handles scalar, numpy scalar, or single-element Series
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)

def _normalize_ticker(t: str) -> str:
    return str(t).strip().replace(".", "-")

def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default

# -------------- S&P 500 LIST (robust) --------------
def get_sp500_tickers(max_retries: int = 3) -> list[str]:
    # FIX: old datahub URL sometimes 404s; use stable GitHub raw CSV
    DATAHUB_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(WIKI_URL, headers=UA, timeout=20)
            resp.raise_for_status()

            tables = pd.read_html(StringIO(resp.text), flavor="lxml")

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

    try:
        resp = requests.get(DATAHUB_CSV, headers=UA, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if "Symbol" not in df.columns:
            raise ValueError("Fallback CSV missing 'Symbol' column.")
        tickers = df["Symbol"].astype(str).map(_normalize_ticker).dropna().unique().tolist()
        if len(tickers) < 450:
            raise ValueError(f"Too few tickers from fallback: {len(tickers)}")
        return tickers
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch S&P 500 tickers. Wiki error: {last_err}; Fallback error: {e}"
        )

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

def normalized_ema_slope(close: pd.Series, length: int = 21, lookback: int = 5) -> float:
    e = ema(close, length)
    if len(e) < lookback + 2:
        return np.nan
    a = e.iloc[-1]
    b = e.iloc[-(lookback + 1)]
    if pd.isna(a) or pd.isna(b) or a == 0:
        return np.nan
    return float((a - b) / a)

def avg_dollar_vol_20(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 25:
        return np.nan
    dv = (df["Close"] * df["Volume"]).rolling(20).mean()
    return _safe_float(dv.iloc[-1], np.nan)

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

# -------------- OPTION SPREAD CHECK (unchanged) --------------
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
    spy = spy.dropna()
    spy["EMA21"] = ema(spy["Close"], SPY_EMA_LEN)

    close_val = _to_float(spy["Close"].iloc[-1])
    ema_val = _to_float(spy["EMA21"].iloc[-1])

    slope_norm = normalized_ema_slope(spy["Close"], SPY_EMA_LEN, SPY_SLOPE_LOOKBACK)
    above = bool(close_val > ema_val)
    below = bool(close_val < ema_val)

    state = "NEUTRAL"
    if above and np.isfinite(slope_norm) and slope_norm > SPY_SLOPE_EPS:
        state = "BULL"
    elif below and np.isfinite(slope_norm) and slope_norm < -SPY_SLOPE_EPS:
        state = "BEAR"

    return {
        "close": close_val,
        "ema21": ema_val,
        "above_ema21": above,
        "slope_norm": float(slope_norm) if np.isfinite(slope_norm) else np.nan,
        "state": state,
    }

# -------------- MAIN SCAN --------------
def scan_inside_days():
    spy_ctx = get_spy_context()
    if spy_ctx:
        print(f"SPY regime: {spy_ctx['state']} | close={spy_ctx['close']:.2f} | ema21={spy_ctx['ema21']:.2f} | slope_norm={spy_ctx['slope_norm']:.6f}")

    try:
        spx = get_sp500_tickers()
    except Exception as e:
        print(f"[FALLBACK] couldn't fetch S&P 500: {e}")
        spx = ["AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA"]

    for etf in MAJOR_ETFS:
        if etf not in spx:
            spx.append(etf)

    if len(spx) < 450:
        raise RuntimeError(f"Unexpected S&P list size (incl. ETFs): {len(spx)} (should be ~500+)")

    print(f"Got {len(spx)} tickers (S&P 500 + major ETFs). Downloading daily bars...")
    ohlc_map = download_daily_spx(spx)
    print(f"Downloaded OK for {len(ohlc_map)} tickers.")

    diag = {
        "tickers_total": 0,
        "len_ge_12": 0,
        "after_intraday_override": 0,
        "atr_ok": 0,
        "inside_ok": 0,
        "compression_ok": 0,
        "passed_all": 0,

        # upgrades diagnostics
        "dvol_failed": 0,
        "atrpct_failed": 0,
        "dist_failed": 0,
        "spy_gate_failed": 0,
        "ticker_align_failed": 0,

        "not_inside_broke_high": 0,
        "not_inside_broke_low": 0,
        "not_inside_other": 0,
        "compression_failed": 0,
    }

    all_inside_days = []

    for t, df in ohlc_map.items():
        diag["tickers_total"] += 1

        if len(df) < 12:
            continue
        diag["len_ge_12"] += 1

        df = override_today_with_intraday(t, df)
        diag["after_intraday_override"] += 1

        df["ATR10"] = atr(df, 10)
        df["EMA21"] = ema(df["Close"], 21)

        last_idx = len(df) - 1
        if last_idx < 1:
            continue

        atr10_val = df["ATR10"].iloc[last_idx]
        if np.isnan(atr10_val) or atr10_val == 0:
            continue
        diag["atr_ok"] += 1

        today_hi = float(df["High"].iloc[last_idx])
        today_lo = float(df["Low"].iloc[last_idx])
        y_hi = float(df["High"].iloc[last_idx - 1])
        y_lo = float(df["Low"].iloc[last_idx - 1])

        if not (today_hi <= y_hi and today_lo >= y_lo):
            if today_hi > y_hi:
                diag["not_inside_broke_high"] += 1
            elif today_lo < y_lo:
                diag["not_inside_broke_low"] += 1
            else:
                diag["not_inside_other"] += 1
            continue
        diag["inside_ok"] += 1

        today = df.iloc[last_idx]
        yesterday = df.iloc[last_idx - 1]

        today_range = today_hi - today_lo

        if today_range > RANGE_COMPRESSION_PCT * float(atr10_val):
            diag["compression_failed"] += 1
            continue
        diag["compression_ok"] += 1

        # ---------------- UPGRADE #2: dollar-volume + ATR% floor ----------------
        dvol20 = avg_dollar_vol_20(df)
        if (not np.isfinite(dvol20)) or dvol20 < MIN_DOLLAR_VOL_20:
            diag["dvol_failed"] += 1
            continue

        close_val = float(today["Close"])
        atr_pct = float(atr10_val) / close_val if close_val else np.nan
        if (not np.isfinite(atr_pct)) or atr_pct < MIN_ATR_PCT:
            diag["atrpct_failed"] += 1
            continue

        # ---------------- UPGRADE #3: distance-to-range (not extended) ----------------
        prev_high = float(yesterday["High"])
        prev_low = float(yesterday["Low"])
        mid = (prev_high + prev_low) / 2.0
        dist_atr = abs(close_val - mid) / float(atr10_val)
        if dist_atr > MAX_DIST_TO_RANGE_ATR:
            diag["dist_failed"] += 1
            continue

        # ---------------- UPGRADE #1: SPY regime gating + ticker alignment ----------------
        ema21_series = df["EMA21"]
        if len(ema21_series) > 5:
            ema21_slope_raw = float(ema21_series.iloc[last_idx] - ema21_series.iloc[last_idx - 5])
        else:
            ema21_slope_raw = float(ema21_series.iloc[last_idx] - ema21_series.iloc[0])

        try:
            ema21_slope_norm = float(ema21_slope_raw) / float(atr10_val)
        except Exception:
            ema21_slope_norm = 0.0

        above_ema = bool(close_val > df["EMA21"].iloc[last_idx]) if USE_TREND_FILTER else True

        # ticker "bias" based on alignment + slope sign
        ticker_bias = "NEUTRAL"
        if above_ema and ema21_slope_norm > 0:
            ticker_bias = "CALL"
        elif (not above_ema) and ema21_slope_norm < 0:
            ticker_bias = "PUT"

        # SPY regime gate (strong only)
        allowed = {"CALL", "PUT"}
        spy_state = spy_ctx["state"] if spy_ctx else "NEUTRAL"
        if spy_state == "BULL":
            allowed = {"CALL"}
        elif spy_state == "BEAR":
            allowed = {"PUT"}

        if spy_state in ("BULL", "BEAR"):
            # require alignment to trade with the regime
            if ticker_bias == "NEUTRAL":
                diag["ticker_align_failed"] += 1
                continue
            if ticker_bias not in allowed:
                diag["spy_gate_failed"] += 1
                continue

        if REQUIRE_TICKER_ALIGN:
            # even in NEUTRAL market, require some alignment (cuts chop signals)
            if ticker_bias == "NEUTRAL":
                diag["ticker_align_failed"] += 1
                continue

        diag["passed_all"] += 1

        compression_ratio = today_range / float(atr10_val)
        compression_score = max(0.0, 1.0 - compression_ratio)
        core_07 = bool(compression_ratio <= CORE_COMPRESSION_PCT)

        open_expansion_score = abs(float(ema21_slope_norm)) * float(compression_score)

        all_inside_days.append(
            {
                "ticker": t,
                "date": df.index[last_idx].date().isoformat(),
                "prev_high": round(prev_high, 4),
                "prev_low": round(prev_low, 4),
                "close": round(close_val, 4),
                "today_range": round(float(today_range), 4),
                "atr10": round(float(atr10_val), 4),

                # upgrades fields
                "dvol20": round(float(dvol20), 2),
                "atr_pct": round(float(atr_pct), 4),
                "dist_atr": round(float(dist_atr), 4),
                "spy_state": spy_state,
                "ticker_bias": ticker_bias,

                "above_ema": above_ema,
                "compression_score": round(float(compression_score), 4),
                "compression_ratio": round(float(compression_ratio), 4),
                "core_07": core_07,
                "ema21_slope": round(float(ema21_slope_raw), 6),
                "ema21_slope_norm": round(float(ema21_slope_norm), 4),
                "ema21_slope_norm_raw": float(ema21_slope_norm),
                "open_expansion_score": round(float(open_expansion_score), 4),
            }
        )

    print("DIAGNOSTICS:")
    print(
        f"  tickers_total={diag['tickers_total']} | len>=12={diag['len_ge_12']} | "
        f"atr_ok={diag['atr_ok']} | inside_ok={diag['inside_ok']} | "
        f"compression_ok={diag['compression_ok']} | passed_all={diag['passed_all']}"
    )
    print(
        f"  upgrade_fails: dvol={diag['dvol_failed']} | atr%={diag['atrpct_failed']} | "
        f"dist={diag['dist_failed']} | spy_gate={diag['spy_gate_failed']} | align={diag['ticker_align_failed']}"
    )
    print(
        f"  not_inside: broke_high={diag['not_inside_broke_high']} | "
        f"broke_low={diag['not_inside_broke_low']} | other={diag['not_inside_other']}"
    )
    print(
        f"  compression_failed={diag['compression_failed']} (threshold={RANGE_COMPRESSION_PCT} * ATR10)"
    )

    print(f"Found {len(all_inside_days)} inside-day candidates.")

    # ---- SCORE + DIRECTION (keep your system, but direction now respects alignment/regime) ----
    filtered = []
    for sig in all_inside_days:
        base = WEIGHT_COMPRESSION * sig["compression_score"]
        if sig["above_ema"]:
            base += WEIGHT_TREND * 1.0

        slope = float(sig.get("ema21_slope_norm_raw", sig["ema21_slope_norm"]))
        slope_bias = _clamp(slope, -1.0, 1.0)
        base += WEIGHT_SLOPE_STRENGTH * abs(slope_bias)

        # Direction: use ticker_bias directly (cleaner than mixing weak components)
        direction = sig.get("ticker_bias", "NEUTRAL")
        if direction not in ("CALL", "PUT"):
            # fallback to your prior combined logic if somehow neutral slips through
            stock_bias = 1.0 if sig["above_ema"] else -1.0
            combined = 0.6 * stock_bias + 0.4 * slope_bias
            direction = "CALL" if combined >= 0 else "PUT"

        # Keep your slope hard-gate (still good)
        if direction == "CALL" and slope < -SLOPE_TOL:
            continue
        if direction == "PUT" and slope > SLOPE_TOL:
            continue

        sig["direction"] = direction
        sig["score"] = round(float(base), 4)
        filtered.append(sig)

    all_inside_days = filtered
    by_ticker = {sig["ticker"]: sig for sig in all_inside_days}

    tight_spread_inside_days = []

    all_inside_days_sorted_for_check = sorted(
        all_inside_days,
        key=lambda x: (x["score"], x.get("open_expansion_score", 0.0)),
        reverse=True,
    )

    for i, sig in enumerate(all_inside_days_sorted_for_check):
        if i >= CHECK_OPTIONS_FOR_MAX:
            break
        best_spread = best_option_spread_for_ticker(sig["ticker"], sig["close"])
        if best_spread is not None:
            base_sig = by_ticker[sig["ticker"]].copy()
            base_sig["best_spread"] = round(float(best_spread), 4)
            base_sig["score"] = round(float(base_sig["score"] + WEIGHT_SPREAD), 4)
            tight_spread_inside_days.append(base_sig)

    all_inside_days = sorted(all_inside_days, key=lambda x: x["score"], reverse=True)
    tight_spread_inside_days = sorted(
        tight_spread_inside_days,
        key=lambda x: (x["score"], x.get("open_expansion_score", 0.0)),
        reverse=True,
    )

    return all_inside_days, tight_spread_inside_days

if __name__ == "__main__":
    all_setups, tight_setups = scan_inside_days()

    tier_a_filtered = [r for r in tight_setups if r["score"] >= 0.55]

    if tier_a_filtered:
        print("\nTIER A (score >= 0.55): inside-day setups WITH tight options")
        print(
            "ticker | score | dir   | spy | core07 | ratio  | date       | prev_high | prev_low | close   | "
            "ATR10 | atr%  | dvol20     | distATR | best_spread | slope_norm | openX"
        )
        for r in tier_a_filtered:
            core_flag = "Y" if r.get("core_07") else "N"
            print(
                f"{r['ticker']:5s} | {r['score']:>5} | {r['direction']:<5} | {str(r.get('spy_state','NA')):>4} | "
                f"{core_flag:>5} | {r.get('compression_ratio','NA'):>5} | {r['date']} | "
                f"{r['prev_high']:>8} | {r['prev_low']:>8} | "
                f"{r['close']:>7} | {r['atr10']:>5} | "
                f"{r.get('atr_pct','NA'):>5} | {r.get('dvol20','NA'):>10} | {r.get('dist_atr','NA'):>6} | "
                f"{r.get('best_spread', 'NA'):>11} | "
                f"{r.get('ema21_slope_norm','NA'):>9} | {r.get('open_expansion_score','NA'):>5}"
            )
    else:
        print("\nTIER A (score >= 0.55): none")
