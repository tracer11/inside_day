#!/usr/bin/env python3
"""
Inside Day Scanner — Expanded Universe Edition v2
Optimized for late-day entry (~3:50pm ET)

Universe includes:
- S&P 500
- Russell 1000 (adds ~500 mid-caps)
- Nasdaq 100 (overlap check)
- Sector ETFs (XLF, XLE, XLK, etc.)
- High-volume non-index names (MSTR, COIN, PLTR, etc.)

v2 Changes:
- Added BLACKLIST for delisted/problematic tickers
- Fixed FutureWarning in get_late_day_momentum()
- Removed VIX (index, not tradeable) from THEMATIC_ETFS
"""

import time
import random
from io import StringIO
from time import sleep

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from curl_cffi import requests as crequests

# –––––––– CONFIG ––––––––

LOOKBACK_DAYS = 30
RANGE_COMPRESSION_PCT = 0.8
CORE_COMPRESSION_PCT = 0.7

USE_TREND_FILTER = True
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
US_MARKET_TZ = "US/Eastern"

# option-spread filters (SOFT: informational only)

CHECK_OPTIONS_FOR_MAX = 120
MAX_ABS_SPREAD = 0.20
MAX_REL_SPREAD = 0.02
ATM_MONEYNES_PCT = 0.02

# scoring weights (SETUP-ONLY) - REVISED FOR LATE-DAY ENTRY

WEIGHT_COMPRESSION = 0.25        # less critical by 3:50pm
WEIGHT_TREND = 0.25              # still matters
WEIGHT_SLOPE_STRENGTH = 0.10  
WEIGHT_BREAK_PROX = 0.25         # CRITICAL - must be near break
WEIGHT_LATE_DAY_DRIFT = 0.15     # momentum in final hour

# Break proximity requirements - TIGHTENED

BREAK_PROX_CAP_ATR = 0.4         # must be within 0.4 ATR
MIN_BREAK_PROX = 0.5             # require break_prox score >= 0.5

# Trend distance scaling cap (in ATR units)

TREND_DIST_CAP_ATR = 1.0

# –––––––– EXPANDED UNIVERSE CONFIG ––––––––

# Major index ETFs
MAJOR_ETFS = ["SPY", "QQQ", "IWM", "DIA"]

# Sector ETFs
SECTOR_ETFS = ["XLF", "XLE", "XLK", "XLV", "XLI", "XLB", "XLC", "XLY", "XLP", "XLU", "XLRE"]

# Thematic / leveraged ETFs with high options volume
# Note: Removed VIX (it's an index, not tradeable - use UVXY/VXX instead)
THEMATIC_ETFS = [
    "ARKK", "ARKG", "ARKW", "ARKF",  # ARK funds
    "SMH", "SOXX",                    # Semiconductors
    "XBI", "IBB",                     # Biotech
    "GDX", "GDXJ", "SLV", "GLD",     # Metals/miners
    "TLT", "HYG", "JNK",             # Bonds
    "EEM", "EWZ", "FXI",             # Intl
    "UVXY", "VXX", "SQQQ", "TQQQ",   # Vol / leveraged (replaced VIX with VXX)
    "XOP", "OIH",                     # Oil & gas
    "KRE", "XHB",                     # Regional banks, homebuilders
]

# High-volume names often not in S&P 500 (memes, crypto-adjacent, growth)
# Note: Removed FSR (delisted)
HIGH_VOL_EXTRAS = [
    # Crypto-adjacent
    "MSTR", "COIN", "HOOD", "MARA", "RIOT", "CLSK", "HUT", "BITF",
    # AI / Tech growth
    "PLTR", "ARM", "SMCI", "IONQ", "RGTI", "QUBT", "AI", "BBAI", "SOUN",
    # EV / Energy (removed FSR - bankrupt)
    "RIVN", "LCID", "NIO", "XPEV", "LI", "GOEV", "CHPT", "BLNK",
    # Fintech / Growth
    "SQ", "AFRM", "UPST", "SOFI", "NU", "SHOP", "SNOW", "DDOG", "NET",
    # Social / Consumer
    "SNAP", "ROKU", "PINS", "RBLX", "U", "TTWO", "EA",
    # Biotech / Pharma (high vol)
    "MRNA", "BNTX", "NVAX",
    # Misc high volume
    "GME", "AMC", "CVNA", "W", "CHWY", "DASH", "ABNB", "UBER", "LYFT",
]

# Tickers known to be delisted, problematic, or non-tradeable
BLACKLIST = {
    "-",        # Bad parse
    "VIX",      # Index, not tradeable
    "BRKB",     # Should be BRK-B
    "BFB",      # Should be BF-B
    "BFA",      # Should be BF-A
    "FSR",      # Fisker - bankrupt/delisted
    "XTSLA",    # Not a real ticker
    "SGAFT",    # Not a real ticker
    "CWENA",    # Clearway Energy - use CWEN
    "HEIA",     # Heineken - not US listed
    "LENB",     # Should be LEN-B
    "UHALB",    # Should be UHAL-B
    "BBBY",     # Bed Bath Beyond - bankrupt/delisted
    "SQ",       # Now trades as XYZ (Block Inc)
    "GOEV",     # Canoo - delisted
}

# –––––––– OTHER CONFIG ––––––––

# EMA slope hard-gate tolerance (normalized by ATR10)

SLOPE_TOL = 0.02

# SPY slope regime gating (BULL/BEAR/NEUTRAL)

SPY_EMA_LEN = 21
SPY_SLOPE_LOOKBACK = 5
SPY_SLOPE_EPS = 0.00015

# SPY close-strength safety valve

SPY_BEAR_NEUTRALIZE_CLOSEPOS = 0.75

# ticker EMA alignment gating

REQUIRE_TICKER_ALIGN = True

# liquidity + movement sanity

MIN_DOLLAR_VOL_20 = 300_000_000
MIN_ATR_PCT = 0.015

# not-extended vs inside-day midpoint

MAX_DIST_TO_RANGE_ATR = 0.60

# regime penalty (soft bias)

REGIME_PENALTY_AGAINST = 0.60

# Late-day momentum requirements

MIN_DRIFT_SCORE = 0.25           # require some directional movement
MIN_VOL_RATIO = 1.1              # volume at least 10% above 20-day avg

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ––––––– UTILS –––––––

def _to_float(x):
    """Safely convert to float, handling Series/DataFrame edge cases"""
    if x is None:
        return np.nan
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)

def _normalize_ticker(t: str) -> str:
    return str(t).strip().replace(".", "-")

def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default

def _close_pos_in_day_range(high: float, low: float, close: float) -> float:
    rng = float(high) - float(low)
    if not np.isfinite(rng) or rng <= 0:
        return np.nan
    return _clamp01((float(close) - float(low)) / rng)

# ––––––– TICKER UNIVERSE FETCHERS –––––––

def get_sp500_tickers(max_retries: int = 3) -> list[str]:
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
        raise RuntimeError(f"Failed to fetch S&P 500 tickers. Wiki error: {last_err}; Fallback error: {e}")


def get_russell_1000_tickers(max_retries: int = 3) -> list[str]:
    """
    Fetch Russell 1000 constituents.
    Primary: ishares holdings CSV
    Fallback: community GitHub mirror
    """
    ISHARES_URL = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
    GITHUB_FALLBACK = "https://raw.githubusercontent.com/datasets/russell-1000-index/main/data/constituents.csv"
    
    last_err = None
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(ISHARES_URL, headers=UA, timeout=30)
            resp.raise_for_status()
            
            lines = resp.text.strip().split('\n')
            header_idx = None
            for i, line in enumerate(lines):
                if 'Ticker' in line:
                    header_idx = i
                    break
            
            if header_idx is None:
                raise ValueError("Could not find Ticker column in iShares CSV")
            
            csv_text = '\n'.join(lines[header_idx:])
            df = pd.read_csv(StringIO(csv_text))
            
            ticker_col = None
            for col in df.columns:
                if 'ticker' in col.lower():
                    ticker_col = col
                    break
            
            if ticker_col is None:
                raise ValueError("Ticker column not found")
            
            tickers = df[ticker_col].astype(str).map(_normalize_ticker).dropna().unique().tolist()
            tickers = [t for t in tickers if t.isalpha() or '-' in t]
            
            if len(tickers) >= 800:
                return tickers
            else:
                raise ValueError(f"Too few tickers from iShares: {len(tickers)}")
                
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                sleep(0.8)
    
    try:
        resp = requests.get(GITHUB_FALLBACK, headers=UA, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        
        col = "Symbol" if "Symbol" in df.columns else ("Ticker" if "Ticker" in df.columns else None)
        if col is None:
            raise ValueError("No ticker column in fallback CSV")
        
        tickers = df[col].astype(str).map(_normalize_ticker).dropna().unique().tolist()
        if len(tickers) >= 500:
            return tickers
        else:
            raise ValueError(f"Too few tickers from GitHub fallback: {len(tickers)}")
            
    except Exception as e:
        print(f"[WARN] Russell 1000 fetch failed. iShares error: {last_err}; GitHub error: {e}")
        return []


def get_nasdaq_100_tickers(max_retries: int = 3) -> list[str]:
    """Fetch Nasdaq 100 as additional coverage"""
    WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(WIKI_NDX, headers=UA, timeout=20)
            resp.raise_for_status()
            
            tables = pd.read_html(StringIO(resp.text), flavor="lxml")
            
            for df in tables:
                cols = {str(c).lower().strip() for c in df.columns}
                if 'ticker' in cols or 'symbol' in cols:
                    sym_col = None
                    for c in df.columns:
                        if 'ticker' in str(c).lower() or 'symbol' in str(c).lower():
                            sym_col = c
                            break
                    if sym_col:
                        tickers = df[sym_col].astype(str).map(_normalize_ticker).dropna().unique().tolist()
                        if len(tickers) >= 90:
                            return tickers
            
            raise ValueError("No suitable table found")
            
        except Exception as e:
            if attempt < max_retries:
                sleep(0.8)
    
    print("[WARN] Nasdaq 100 fetch failed")
    return []


def build_expanded_universe() -> list[str]:
    """Combine all ticker sources into deduplicated universe"""
    print("Building expanded ticker universe...")
    
    spx = get_sp500_tickers()
    print(f"  S&P 500: {len(spx)} tickers")
    
    russell = get_russell_1000_tickers()
    print(f"  Russell 1000: {len(russell)} tickers")
    
    ndx = get_nasdaq_100_tickers()
    print(f"  Nasdaq 100: {len(ndx)} tickers")
    
    # Combine all sources
    all_tickers = set(spx)
    all_tickers.update(russell)
    all_tickers.update(ndx)
    all_tickers.update(MAJOR_ETFS)
    all_tickers.update(SECTOR_ETFS)
    all_tickers.update(THEMATIC_ETFS)
    all_tickers.update(HIGH_VOL_EXTRAS)
    
    # Remove blacklisted tickers
    all_tickers -= BLACKLIST
    
    # Clean up invalid entries
    all_tickers = {t for t in all_tickers if t and len(t) <= 5 and t not in ('', 'nan', 'None')}
    
    all_tickers = sorted(all_tickers)
    print(f"  TOTAL UNIVERSE: {len(all_tickers)} unique tickers (after blacklist)")
    
    return all_tickers


# ––––––– TECHNICALS –––––––

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

def normalized_ema_slope(close_like, length: int = 21, lookback: int = 5) -> float:
    if isinstance(close_like, pd.DataFrame):
        close = close_like.iloc[:, 0]
    else:
        close = close_like

    close = pd.Series(close).dropna()
    if len(close) < lookback + 2:
        return np.nan

    e = ema(close, length)

    a = e.iloc[-1]
    b = e.iloc[-(lookback + 1)]

    a = _to_float(a)
    b = _to_float(b)

    if np.isnan(a) or np.isnan(b) or a == 0:
        return np.nan

    return float((a - b) / a)

def avg_dollar_vol_20(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 25:
        return np.nan
    dv = (df["Close"] * df["Volume"]).rolling(20).mean()
    return _safe_float(dv.iloc[-1], np.nan)


# ––––––– LATE-DAY METRICS –––––––

def get_late_day_momentum(intraday_df, direction: str, prev_high: float, prev_low: float, close: float) -> dict:
    """Check if last hour shows directional bias toward breakout"""
    if intraday_df is None or len(intraday_df) < 6:
        return {
            "drift_score": 0.0,
            "drift_ok": False,
            "close_pos_intraday": np.nan,
            "structure_ok": False
        }

    # Last hour momentum (12x 5-min bars)
    last_hour = intraday_df.iloc[-min(12, len(intraday_df)):]

    # Use _to_float to avoid FutureWarning
    close_1hr_ago = _to_float(last_hour['Close'].iloc[0])
    close_now = _to_float(last_hour['Close'].iloc[-1])

    # Calculate drift toward expected breakout direction
    if direction == "CALL":
        if close_1hr_ago < prev_high:
            drift = (close_now - close_1hr_ago) / (prev_high - close_1hr_ago)
        else:
            drift = 0.0
    else:  # PUT
        if close_1hr_ago > prev_low:
            drift = (close_1hr_ago - close_now) / (close_1hr_ago - prev_low)
        else:
            drift = 0.0

    drift_score = max(0.0, min(1.0, drift))

    # Intraday structure check - use _to_float to avoid FutureWarning
    intraday_high = _to_float(intraday_df['High'].max())
    intraday_low = _to_float(intraday_df['Low'].min())
    intraday_range = intraday_high - intraday_low

    if intraday_range > 0:
        close_position = (close - intraday_low) / intraday_range
    else:
        close_position = 0.5

    # Structure validation: for calls want to be near highs, puts near lows
    if direction == "CALL":
        structure_ok = (close_position > 0.6 and 
                       (prev_high - close) < intraday_range * 0.5)
    else:  # PUT
        structure_ok = (close_position < 0.4 and 
                       (close - prev_low) < intraday_range * 0.5)

    return {
        "drift_score": round(float(drift_score), 4),
        "drift_ok": drift_score > MIN_DRIFT_SCORE,
        "close_pos_intraday": round(float(close_position), 3),
        "structure_ok": bool(structure_ok)
    }

def volume_pattern_ok(df: pd.DataFrame) -> dict:
    """Check if today's volume is above recent average"""
    if len(df) < 20:
        return {"vol_ratio": 1.0, "vol_surge": False}

    avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
    today_vol = _to_float(df['Volume'].iloc[-1])

    vol_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

    return {
        "vol_ratio": round(float(vol_ratio), 2),
        "vol_surge": vol_ratio >= MIN_VOL_RATIO
    }


# ––––––– DAILY DOWNLOAD (batch + curl_cffi session) –––––––

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

def download_daily_data(tickers: list[str]) -> dict:
    """Download daily OHLCV for all tickers"""
    sess = crequests.Session(impersonate="chrome")

    data = {}
    failed = []

    BATCH_SIZE = 50
    MAX_RETRIES = 4
    TIMEOUT_S = 30

    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

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
        print(f"  batch {batch_idx}/{total_batches}: downloaded {len(data)} / {done} tickers")
        time.sleep(0.6 + random.random() * 0.6)

    if failed:
        print(f"\n[WARN] Failed downloads ({len(set(failed))}): {sorted(set(failed))[:20]}...")

    return data


# ––––––– INTRADAY OVERRIDE + RETURN INTRADAY DF –––––––

def override_today_with_intraday(ticker: str, df_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Returns (updated_daily, intraday_df)"""
    if df_daily is None or df_daily.empty:
        return df_daily, None

    now_et = pd.Timestamp.now(tz=US_MARKET_TZ)
    today_date = now_et.date()

    last_ts = df_daily.index[-1]
    last_date = last_ts.date()

    if last_date != today_date:
        return df_daily, None

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
        return df_daily, None

    if intraday is None or intraday.empty:
        return df_daily, None

    if intraday.index.tz is None:
        intraday = intraday.tz_localize("UTC").tz_convert(US_MARKET_TZ)
    else:
        intraday = intraday.tz_convert(US_MARKET_TZ)

    intraday_today = intraday[intraday.index.date == today_date]
    if intraday_today.empty:
        return df_daily, None

    hi = _to_float(intraday_today["High"].max())
    lo = _to_float(intraday_today["Low"].min())
    close = _to_float(intraday_today["Close"].iloc[-1])

    df_daily.iloc[-1, df_daily.columns.get_loc("High")] = hi
    df_daily.iloc[-1, df_daily.columns.get_loc("Low")] = lo
    df_daily.iloc[-1, df_daily.columns.get_loc("Close")] = close

    return df_daily, intraday_today


# ––––––– OPTION SPREAD CHECK (SOFT, informational only) –––––––

def best_option_spread_for_ticker(ticker: str, spot: float):
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None
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
            if float(bid) <= 0:
                continue
            if spot == 0:
                continue
            if abs(float(strike) - float(spot)) / float(spot) > ATM_MONEYNES_PCT:
                continue

            spread = float(ask) - float(bid)
            if spread <= 0:
                continue

            if best_spread is None or spread < best_spread:
                best_spread = spread

    for exp in (yf.Ticker(ticker).options or [])[:2]:
        try:
            chain = yf.Ticker(ticker).option_chain(exp)
        except Exception:
            continue
        scan_df(chain.calls if hasattr(chain, "calls") else None)
        scan_df(chain.puts if hasattr(chain, "puts") else None)

    return best_spread

def spread_ok(best_spread: float | None, est_mid: float | None = None) -> bool:
    if best_spread is None:
        return False
    return float(best_spread) <= float(MAX_ABS_SPREAD)


# ––––––– MARKET CONTEXT (SPY) –––––––

def get_spy_context():
    spy = yf.download("SPY", period="90d", interval="1d", progress=False, auto_adjust=False, timeout=30)
    if spy is None or spy.empty:
        return None
    spy = spy.dropna()
    spy["EMA21"] = ema(spy["Close"], SPY_EMA_LEN)

    close_val = _to_float(spy["Close"].iloc[-1])
    ema_val = _to_float(spy["EMA21"].iloc[-1])
    hi_val = _to_float(spy["High"].iloc[-1])
    lo_val = _to_float(spy["Low"].iloc[-1])

    close_series = spy["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    slope_norm = normalized_ema_slope(close_series, SPY_EMA_LEN, SPY_SLOPE_LOOKBACK)
    above = bool(close_val > ema_val)
    below = bool(close_val < ema_val)

    close_pos = _close_pos_in_day_range(hi_val, lo_val, close_val)

    state = "NEUTRAL"
    if above and np.isfinite(slope_norm) and slope_norm > SPY_SLOPE_EPS:
        state = "BULL"
    elif below and np.isfinite(slope_norm) and slope_norm < -SPY_SLOPE_EPS:
        state = "BEAR"

    if state == "BEAR" and np.isfinite(close_pos) and close_pos > SPY_BEAR_NEUTRALIZE_CLOSEPOS:
        state = "NEUTRAL"

    return {
        "close": close_val,
        "high": hi_val,
        "low": lo_val,
        "close_pos": float(close_pos) if np.isfinite(close_pos) else np.nan,
        "ema21": ema_val,
        "above_ema21": above,
        "slope_norm": float(slope_norm) if np.isfinite(slope_norm) else np.nan,
        "state": state,
    }


# ––––––– MAIN SCAN –––––––

def scan_inside_days():
    spy_ctx = get_spy_context()
    if spy_ctx:
        print(
            f"SPY regime: {spy_ctx['state']} | close={spy_ctx['close']:.2f} | "
            f"ema21={spy_ctx['ema21']:.2f} | slope_norm={spy_ctx['slope_norm']:.6f} | "
            f"close_pos={spy_ctx['close_pos']:.3f}"
        )

    # Build expanded universe
    all_tickers = build_expanded_universe()

    print(f"\nDownloading daily bars for {len(all_tickers)} tickers...")
    ohlc_map = download_daily_data(all_tickers)
    print(f"Downloaded OK for {len(ohlc_map)} tickers.\n")

    diag = {
        "tickers_total": 0,
        "len_ge_12": 0,
        "after_intraday_override": 0,
        "atr_ok": 0,
        "inside_ok": 0,
        "compression_ok": 0,
        "passed_all": 0,

        "dvol_failed": 0,
        "atrpct_failed": 0,
        "dist_failed": 0,
        "ticker_align_failed": 0,
        "against_regime": 0,
        
        "drift_failed": 0,
        "vol_failed": 0,
        "structure_failed": 0,
        "break_prox_failed": 0,

        "options_checked": 0,
        "spread_ok": 0,
        "spread_none": 0,

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

        df, intraday_df = override_today_with_intraday(t, df)
        diag["after_intraday_override"] += 1

        df["ATR10"] = atr(df, 10)
        df["EMA21"] = ema(df["Close"], 21)

        last_idx = len(df) - 1
        if last_idx < 1:
            continue

        atr10_val = df["ATR10"].iloc[last_idx]
        if np.isnan(_to_float(atr10_val)) or _to_float(atr10_val) == 0:
            continue
        diag["atr_ok"] += 1

        today_hi = _to_float(df["High"].iloc[last_idx])
        today_lo = _to_float(df["Low"].iloc[last_idx])
        y_hi = _to_float(df["High"].iloc[last_idx - 1])
        y_lo = _to_float(df["Low"].iloc[last_idx - 1])

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
        atr10_val = _to_float(atr10_val)

        if today_range > RANGE_COMPRESSION_PCT * atr10_val:
            diag["compression_failed"] += 1
            continue
        diag["compression_ok"] += 1

        dvol20 = avg_dollar_vol_20(df)
        if (not np.isfinite(dvol20)) or dvol20 < MIN_DOLLAR_VOL_20:
            diag["dvol_failed"] += 1
            continue

        close_val = _to_float(today["Close"])
        atr_pct = atr10_val / close_val if close_val else np.nan
        if (not np.isfinite(atr_pct)) or atr_pct < MIN_ATR_PCT:
            diag["atrpct_failed"] += 1
            continue

        prev_high = _to_float(yesterday["High"])
        prev_low = _to_float(yesterday["Low"])
        mid = (prev_high + prev_low) / 2.0
        dist_atr = abs(close_val - mid) / atr10_val
        if dist_atr > MAX_DIST_TO_RANGE_ATR:
            diag["dist_failed"] += 1
            continue

        ema21_series = df["EMA21"]
        if len(ema21_series) > 5:
            ema21_slope_raw = _to_float(ema21_series.iloc[last_idx]) - _to_float(ema21_series.iloc[last_idx - 5])
        else:
            ema21_slope_raw = _to_float(ema21_series.iloc[last_idx]) - _to_float(ema21_series.iloc[0])

        try:
            ema21_slope_norm = float(ema21_slope_raw) / atr10_val
        except Exception:
            ema21_slope_norm = 0.0

        ema21_val = _to_float(df["EMA21"].iloc[last_idx])
        above_ema = bool(close_val > ema21_val) if USE_TREND_FILTER else True

        ticker_bias = "NEUTRAL"
        if above_ema and ema21_slope_norm > 0:
            ticker_bias = "CALL"
        elif (not above_ema) and ema21_slope_norm < 0:
            ticker_bias = "PUT"

        if REQUIRE_TICKER_ALIGN and ticker_bias == "NEUTRAL":
            diag["ticker_align_failed"] += 1
            continue

        spy_state = spy_ctx["state"] if spy_ctx else "NEUTRAL"
        against_regime = False
        if spy_state == "BULL" and ticker_bias == "PUT":
            against_regime = True
        elif spy_state == "BEAR" and ticker_bias == "CALL":
            against_regime = True

        if against_regime:
            diag["against_regime"] += 1

        # Volume check
        vol_metrics = volume_pattern_ok(df)
        if not vol_metrics["vol_surge"]:
            diag["vol_failed"] += 1
            continue

        diag["passed_all"] += 1

        compression_ratio = today_range / atr10_val
        compression_score = max(0.0, 1.0 - compression_ratio)
        core_07 = bool(compression_ratio <= CORE_COMPRESSION_PCT)

        open_expansion_score = abs(float(ema21_slope_norm)) * float(compression_score)

        trend_dist_atr_raw = abs(close_val - ema21_val) / atr10_val
        trend_strength = _clamp01(trend_dist_atr_raw / float(TREND_DIST_CAP_ATR))

        dist_to_prev_low_atr = (close_val - prev_low) / atr10_val
        dist_to_prev_high_atr = (prev_high - close_val) / atr10_val

        all_inside_days.append(
            {
                "ticker": t,
                "date": df.index[last_idx].date().isoformat(),
                "prev_high": round(prev_high, 4),
                "prev_low": round(prev_low, 4),
                "close": round(close_val, 4),
                "ema21": round(ema21_val, 4),

                "today_range": round(float(today_range), 4),
                "atr10": round(float(atr10_val), 4),

                "dvol20": round(float(dvol20), 2),
                "atr_pct": round(float(atr_pct), 4),
                "dist_atr": round(float(dist_atr), 4),
                "spy_state": spy_state,
                "ticker_bias": ticker_bias,
                "against_regime": bool(against_regime),

                "above_ema": above_ema,
                "compression_score": round(float(compression_score), 4),
                "compression_ratio": round(float(compression_ratio), 4),
                "core_07": core_07,

                "ema21_slope": round(float(ema21_slope_raw), 6),
                "ema21_slope_norm": round(float(ema21_slope_norm), 4),
                "ema21_slope_norm_raw": float(ema21_slope_norm),

                "trend_dist_atr": round(float(trend_dist_atr_raw), 4),
                "trend_strength": round(float(trend_strength), 4),

                "dist_to_prev_low_atr": round(float(dist_to_prev_low_atr), 4),
                "dist_to_prev_high_atr": round(float(dist_to_prev_high_atr), 4),

                "open_expansion_score": round(float(open_expansion_score), 4),
                
                # Volume metrics
                "vol_ratio": vol_metrics["vol_ratio"],
                "vol_surge": vol_metrics["vol_surge"],
                
                # Intraday data for drift calc
                "_intraday_df": intraday_df,
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
        f"dist={diag['dist_failed']} | align={diag['ticker_align_failed']} | "
        f"vol={diag['vol_failed']} | against_regime={diag['against_regime']}"
    )
    print(
        f"  not_inside: broke_high={diag['not_inside_broke_high']} | "
        f"broke_low={diag['not_inside_broke_low']} | other={diag['not_inside_other']}"
    )
    print(f"  compression_failed={diag['compression_failed']} (threshold={RANGE_COMPRESSION_PCT} * ATR10)")
    print(f"Found {len(all_inside_days)} inside-day candidates.")

    # ---- SCORE + DIRECTION WITH LATE-DAY DRIFT ----
    filtered = []
    for sig in all_inside_days:
        direction = sig.get("ticker_bias", "NEUTRAL")
        if direction not in ("CALL", "PUT"):
            stock_bias = 1.0 if sig["above_ema"] else -1.0
            slope = float(sig.get("ema21_slope_norm_raw", sig["ema21_slope_norm"]))
            slope_bias = _clamp(slope, -1.0, 1.0)
            combined = 0.6 * stock_bias + 0.4 * slope_bias
            direction = "CALL" if combined >= 0 else "PUT"

        # Slope sanity gate
        slope = float(sig.get("ema21_slope_norm_raw", sig["ema21_slope_norm"]))
        if direction == "CALL" and slope < -SLOPE_TOL:
            continue
        if direction == "PUT" and slope > SLOPE_TOL:
            continue

        # Calculate late-day momentum
        intraday_df = sig.pop("_intraday_df", None)
        drift_metrics = get_late_day_momentum(
            intraday_df,
            direction,
            sig["prev_high"],
            sig["prev_low"],
            sig["close"]
        )
        
        # HARD FILTER: require drift
        if not drift_metrics["drift_ok"]:
            diag["drift_failed"] += 1
            continue
        
        # HARD FILTER: require structure
        if not drift_metrics["structure_ok"]:
            diag["structure_failed"] += 1
            continue

        # Calculate base score with new weights
        base = WEIGHT_COMPRESSION * sig["compression_score"]

        trend_strength = float(sig.get("trend_strength", 0.0))
        base += WEIGHT_TREND * _clamp01(trend_strength)

        slope_bias = _clamp(slope, -1.0, 1.0)
        base += WEIGHT_SLOPE_STRENGTH * abs(slope_bias)

        # Break proximity score
        atr10_val = float(sig["atr10"])
        prev_high = float(sig["prev_high"])
        prev_low = float(sig["prev_low"])
        close_val = float(sig["close"])

        if atr10_val > 0:
            if direction == "PUT":
                dist_atr = (close_val - prev_low) / atr10_val
            else:  # CALL
                dist_atr = (prev_high - close_val) / atr10_val

            break_prox = 1.0 - _clamp01(dist_atr / float(BREAK_PROX_CAP_ATR))
        else:
            break_prox = 0.0
            dist_atr = np.nan

        # HARD FILTER: require minimum break proximity
        if break_prox < MIN_BREAK_PROX:
            diag["break_prox_failed"] += 1
            continue

        base += WEIGHT_BREAK_PROX * _clamp01(break_prox)
        
        # Add late-day drift score
        base += WEIGHT_LATE_DAY_DRIFT * drift_metrics["drift_score"]

        # Regime penalty
        spy_state = sig.get("spy_state", "NEUTRAL")
        against = False
        if spy_state == "BULL" and direction == "PUT":
            against = True
        elif spy_state == "BEAR" and direction == "CALL":
            against = True

        penalty = REGIME_PENALTY_AGAINST if against else 1.0
        base *= float(penalty)

        sig["direction"] = direction
        sig["regime_penalty"] = round(float(penalty), 3)
        sig["score"] = round(float(base), 4)
        sig["break_prox"] = round(float(break_prox), 4)
        sig["break_dist_atr"] = round(float(dist_atr), 4) if atr10_val > 0 else np.nan
        
        # Add drift metrics to output
        sig["drift_score"] = drift_metrics["drift_score"]
        sig["drift_ok"] = drift_metrics["drift_ok"]
        sig["close_pos_intraday"] = drift_metrics["close_pos_intraday"]
        sig["structure_ok"] = drift_metrics["structure_ok"]
        
        filtered.append(sig)

    all_inside_days = filtered
    by_ticker = {sig["ticker"]: sig for sig in all_inside_days}

    print(f"  late_day_fails: drift={diag['drift_failed']} | structure={diag['structure_failed']} | break_prox={diag['break_prox_failed']}")

    # Spread is SOFT informational only
    scored_with_spread = []

    all_inside_days_sorted_for_check = sorted(
        all_inside_days,
        key=lambda x: (x["score"], x.get("open_expansion_score", 0.0)),
        reverse=True,
    )

    for i, sig in enumerate(all_inside_days_sorted_for_check):
        if i >= CHECK_OPTIONS_FOR_MAX:
            break

        diag["options_checked"] += 1
        bs = best_option_spread_for_ticker(sig["ticker"], sig["close"])
        if bs is None:
            diag["spread_none"] += 1
        ok = spread_ok(bs)
        if ok:
            diag["spread_ok"] += 1

        base_sig = by_ticker[sig["ticker"]].copy()
        base_sig["best_spread"] = round(float(bs), 4) if bs is not None else None
        base_sig["spread_ok"] = bool(ok)

        scored_with_spread.append(base_sig)

    all_inside_days = sorted(all_inside_days, key=lambda x: x["score"], reverse=True)
    scored_with_spread = sorted(
        scored_with_spread,
        key=lambda x: (x["score"], x.get("open_expansion_score", 0.0)),
        reverse=True,
    )

    print(f"OPTIONS (soft): checked={diag['options_checked']} | spread_ok={diag['spread_ok']} | spread_none={diag['spread_none']}")
    return all_inside_days, scored_with_spread


if __name__ == "__main__":
    all_setups, setups_scored_with_spread = scan_inside_days()

    tier_a_filtered = [r for r in setups_scored_with_spread if r["score"] >= 0.55]

    if tier_a_filtered:
        print("\n" + "="*160)
        print("TIER A (score >= 0.55): LATE-DAY ENTRY SETUPS (optimized for 3:50pm ET entry)")
        print("="*160)
        print(
            "ticker | score | dir  | spy | pen | drift | struc | core07 | ratio | date       | prev_high | prev_low | close   | "
            "EMA21 | tDist | ATR10 | atr% | dvol20     | bProx | bDist | volR | spread_ok | best_spread | slope_norm | openX"
        )
        print("-"*160)
        for r in tier_a_filtered:
            core_flag = "Y" if r.get("core_07") else "N"
            drift_flag = "Y" if r.get("drift_ok") else "N"
            struc_flag = "Y" if r.get("structure_ok") else "N"
            sp_ok = "Y" if r.get("spread_ok") else "N"
            bs = r.get("best_spread")
            bs_str = f"{bs:>10.4f}" if isinstance(bs, (int, float)) else f"{'NA':>10}"
            print(
                f"{r['ticker']:6s} | {r['score']:>5.3f} | {r['direction']:<4} | {str(r.get('spy_state','NA')):>3} | "
                f"{r.get('regime_penalty','NA'):>3.2f} | {drift_flag:>5} | {struc_flag:>5} | {core_flag:>6} | {r.get('compression_ratio','NA'):>5.3f} | "
                f"{r['date']} | {r['prev_high']:>9.2f} | {r['prev_low']:>8.2f} | {r['close']:>7.2f} | "
                f"{r.get('ema21','NA'):>5.2f} | {r.get('trend_dist_atr','NA'):>5.3f} | {r['atr10']:>5.2f} | "
                f"{r.get('atr_pct','NA'):>4.3f} | {r.get('dvol20','NA'):>10,.0f} | "
                f"{r.get('break_prox','NA'):>5.3f} | {r.get('break_dist_atr','NA'):>5.3f} | "
                f"{r.get('vol_ratio','NA'):>4.2f} | {sp_ok:>9} | {bs_str} | {r.get('ema21_slope_norm','NA'):>10.4f} | {r.get('open_expansion_score','NA'):>5.3f}"
            )
        print("="*160)
        print(f"\nKEY FILTERS FOR 3:50PM ENTRY:")
        print(f"  • drift_ok: Late-day momentum toward breakout direction (min={MIN_DRIFT_SCORE})")
        print(f"  • struc: Intraday structure validates setup (calls near highs, puts near lows)")
        print(f"  • bProx: Break proximity tightened (must be within {BREAK_PROX_CAP_ATR} ATR, min score {MIN_BREAK_PROX})")
        print(f"  • volR: Volume ratio vs 20-day avg (min={MIN_VOL_RATIO})")
        print(f"  • All setups passed: drift, structure, break-proximity, and volume filters")
    else:
        print("\nTIER A (score >= 0.55): none")
        print("No setups passed the late-day momentum and structure filters.")

    # Also show Tier B for reference
    tier_b = [r for r in setups_scored_with_spread if r["score"] < 0.55]
    if tier_b:
        print(f"\nTIER B (score < 0.55): {len(tier_b)} setups — lower conviction, consider smaller size or skip")