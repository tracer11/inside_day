#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from time import sleep

# ---------------- CONFIG ----------------
LOOKBACK_DAYS = 30
RANGE_COMPRESSION_PCT = 0.7
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

# Major ETFs explicitly included in the scan
MAJOR_ETFS = ["SPY", "QQQ", "IWM", "DIA"]

# Download batching
DOWNLOAD_BATCH_SIZE = 50
DOWNLOAD_BATCH_PAUSE = 0.7
DOWNLOAD_RETRIES = 2

# Sector filter (MODE 1)
USE_SECTOR_FILTER = True

# Official GICS Sector -> SPDR Sector ETF mapping
GICS_TO_SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# -------------- UTILS --------------
def _to_float(x):
    """
    Robust scalar extractor:
    - handles numpy scalars
    - handles pandas Series (1-element or more)
    - handles pandas DataFrame (1x1) if it ever slips through
    """
    if x is None:
        return np.nan

    # numpy scalar / pandas scalar
    if hasattr(x, "item") and not hasattr(x, "iloc"):
        try:
            return float(x.item())
        except Exception:
            pass

    # pandas Series/DataFrame
    if hasattr(x, "iloc"):
        try:
            v = x.iloc[-1]
            if hasattr(v, "item"):
                return float(v.item())
            return float(v)
        except Exception:
            return float(np.asarray(x).ravel()[-1])

    return float(x)

def _normalize_ticker(t: str) -> str:
    # Yahoo uses '-' instead of '.' (e.g., BRK.B -> BRK-B)
    return str(t).strip().replace(".", "-")

# -------------- S&P 500 LIST + SECTORS (robust) --------------
def get_sp500_with_sectors(max_retries: int = 3):
    """
    Fetch S&P 500 constituents from Wikipedia INCLUDING GICS Sector.
    Returns: (tickers_list, ticker_to_sector_dict)

    If Wikipedia fails, fallback to Datahub for tickers only (sectors unavailable -> empty dict).
    """
    DATAHUB_CSV = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    last_err = None

    # Primary: Wikipedia with GICS Sector
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(WIKI_URL, headers=UA, timeout=20)
            resp.raise_for_status()
            # FIX: wrap literal HTML string with StringIO (pandas deprecation)
            tables = pd.read_html(StringIO(resp.text))

            candidates = []
            for df in tables:
                cols = {str(c).lower().strip() for c in df.columns}
                # Require symbol + GICS sector to be present
                if ("symbol" in cols or "ticker symbol" in cols) and ("gics sector" in cols):
                    candidates.append(df)

            if not candidates:
                raise ValueError("No S&P table with 'GICS Sector' found on Wikipedia.")

            df = max(candidates, key=len)

            sym_col = "Symbol" if "Symbol" in df.columns else "Ticker symbol"
            sector_col = "GICS Sector" if "GICS Sector" in df.columns else "GICS sector"

            tickers = (
                df[sym_col]
                .astype(str)
                .map(_normalize_ticker)
                .dropna()
                .unique()
                .tolist()
            )

            if len(tickers) < 450:
                raise ValueError(f"Too few tickers parsed from Wikipedia: {len(tickers)}")

            ticker_to_sector = {}
            for sym, sec in zip(df[sym_col], df[sector_col]):
                t = _normalize_ticker(sym)
                s = str(sec).strip()
                if t and s and s != "nan":
                    ticker_to_sector[t] = s

            return tickers, ticker_to_sector

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                sleep(0.8)
            else:
                break

    # Fallback: Datahub tickers only (no sectors)
    try:
        resp = requests.get(DATAHUB_CSV, headers=UA, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if "Symbol" not in df.columns:
            raise ValueError("Fallback CSV missing 'Symbol' column.")
        tickers = df["Symbol"].astype(str).map(_normalize_ticker).dropna().unique().tolist()
        if len(tickers) < 450:
            raise ValueError(f"Too few tickers from fallback: {len(tickers)}")
        return tickers, {}
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch S&P 500. Wiki error: {last_err}; Fallback error: {e}"
        )

# -------------- TECHNICALS --------------
def atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(index=df.index if df is not None else None, dtype=float)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    parts = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )

    if parts.empty:
        return pd.Series(index=df.index, dtype=float)

    tr = parts.max(axis=1, skipna=True)
    return tr.rolling(period).mean()

def ema(series: pd.Series, length: int = 21) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def is_inside_day(df: pd.DataFrame, idx: int) -> bool:
    return (
        df["High"].iloc[idx] <= df["High"].iloc[idx - 1]
        and df["Low"].iloc[idx] >= df["Low"].iloc[idx - 1]
    )

# -------------- DAILY DOWNLOAD --------------
def download_one_ticker(ticker: str, session: requests.Session, period="60d"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            session=session,
        )
    except TypeError:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        df = None

    if df is None or df.empty:
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval="1d", auto_adjust=False)
        except Exception:
            df = None

    if df is None or df.empty:
        return None

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        return None
    return df

def download_daily_spx(
    tickers,
    batch_size: int = DOWNLOAD_BATCH_SIZE,
    pause: float = DOWNLOAD_BATCH_PAUSE,
    retries: int = DOWNLOAD_RETRIES,
):
    """
    Batched + throttled downloader to reduce Yahoo timeouts/rate-limits.
    """
    sess = requests.Session()
    sess.headers.update(UA)
    data = {}

    total = len(tickers)
    batches = [tickers[i:i + batch_size] for i in range(0, total, batch_size)]

    for bi, batch in enumerate(batches, 1):
        print(f"...batch {bi}: downloading {len(batch)} tickers")

        for t in batch:
            df = None
            last_err = None

            for attempt in range(retries + 1):
                try:
                    df = download_one_ticker(t, sess, period=f"{LOOKBACK_DAYS}d")
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    last_err = e
                sleep(0.15)

            if df is None or df.empty:
                if last_err:
                    print(f"[WARN] no data for {t} (err={last_err})")
                else:
                    print(f"[WARN] no data for {t}")
                continue

            data[t] = df

        if bi < len(batches):
            sleep(pause)

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

    hi_val = intraday_today["High"].max()
    lo_val = intraday_today["Low"].min()
    close_val = intraday_today["Close"].iloc[-1]

    hi = _to_float(hi_val)
    lo = _to_float(lo_val)
    close = _to_float(close_val)

    high_col = df_daily.columns.get_loc("High")
    low_col = df_daily.columns.get_loc("Low")
    close_col = df_daily.columns.get_loc("Close")

    df_daily.iloc[-1, high_col] = hi
    df_daily.iloc[-1, low_col] = lo
    df_daily.iloc[-1, close_col] = close

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
    spy = yf.download("SPY", period="60d", interval="1d", progress=False, auto_adjust=False)
    if spy is None or spy.empty:
        return None
    spy["EMA21"] = ema(spy["Close"], 21)
    # FIX: avoid float(Series) deprecation
    close_val = _to_float(spy["Close"].iloc[-1])
    ema_val = _to_float(spy["EMA21"].iloc[-1])
    return {
        "close": close_val,
        "above_ema21": bool(close_val > ema_val)
    }

# -------------- SECTOR CONTEXT --------------
def get_sector_context():
    """
    Download each sector ETF once and compute above/below EMA21.
    Returns dict: etf -> {"close": float, "above_ema21": bool}
    """
    ctx = {}
    etfs = sorted(set(GICS_TO_SECTOR_ETF.values()))
    for etf in etfs:
        df = yf.download(etf, period="60d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            continue
        df["EMA21"] = ema(df["Close"], 21)
        # FIX: avoid float(Series) deprecation
        close_val = _to_float(df["Close"].iloc[-1])
        ema_val = _to_float(df["EMA21"].iloc[-1])
        ctx[etf] = {"close": close_val, "above_ema21": bool(close_val > ema_val)}
    return ctx

def sector_ok_for_trade(ticker: str, direction: str, ticker_to_sector: dict, sector_ctx: dict) -> tuple[bool, str]:
    """
    MODE 1:
      CALL allowed only if sector ETF above EMA21.
      PUT allowed only if sector ETF below EMA21.
    Returns: (ok, sector_etf_or_reason)
    """
    sector = ticker_to_sector.get(ticker)
    if not sector:
        return True, "NA"  # fail-open if sector unknown (should be rare)

    etf = GICS_TO_SECTOR_ETF.get(sector)
    if not etf:
        return True, "NA"  # fail-open if not mapped

    ctx = sector_ctx.get(etf)
    if not ctx:
        return True, etf  # fail-open if ETF data missing

    above = bool(ctx["above_ema21"])
    if direction == "CALL":
        return (above, etf)
    if direction == "PUT":
        return ((not above), etf)
    return True, etf

# -------------- MAIN SCAN --------------
def scan_inside_days():
    spy_ctx = get_spy_context()
    sector_ctx = get_sector_context() if USE_SECTOR_FILTER else {}

    try:
        spx, ticker_to_sector = get_sp500_with_sectors()
    except Exception as e:
        print(f"[FALLBACK] couldn't fetch S&P 500 with sectors: {e}")
        spx = ["AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA"]
        ticker_to_sector = {}

    # Add major ETFs explicitly (these are NOT in S&P GICS sectors; we treat them as always sector-ok)
    for etf in MAJOR_ETFS:
        if etf not in spx:
            spx.append(etf)

    if len(spx) < 450:
        raise RuntimeError(f"Unexpected S&P list size (incl. ETFs): {len(spx)} (should be ~500+)")

    print(f"Got {len(spx)} tickers (S&P 500 + major ETFs). Downloading daily bars...")
    ohlc_map = download_daily_spx(spx)
    print(f"Downloaded OK for {len(ohlc_map)} tickers.")
    print(f"Missing tickers: {len(spx) - len(ohlc_map)}")

    all_inside_days = []

    for t, df in ohlc_map.items():
        if len(df) < 12:
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

        today_range = today["High"] - today["Low"]
        atr10 = df["ATR10"].iloc[last_idx]
        if np.isnan(atr10) or atr10 == 0:
            continue
        if today_range > RANGE_COMPRESSION_PCT * atr10:
            continue

        if USE_TREND_FILTER:
            above_ema = bool(today["Close"] > df["EMA21"].iloc[last_idx])
        else:
            above_ema = True

        # EMA21 slope over 5 trading days
        ema21_series = df["EMA21"]
        if len(ema21_series) > 5:
            ema21_slope = float(ema21_series.iloc[last_idx] - ema21_series.iloc[last_idx - 5])
        else:
            ema21_slope = float(ema21_series.iloc[last_idx] - ema21_series.iloc[0])

        compression_ratio = today_range / atr10
        compression_score = max(0.0, 1.0 - compression_ratio)

        market_ok = True
        if spy_ctx is not None and not spy_ctx["above_ema21"]:
            market_ok = False

        all_inside_days.append(
            {
                "ticker": t,
                "date": df.index[last_idx].date().isoformat(),
                "prev_high": round(float(yesterday["High"]), 4),
                "prev_low": round(float(yesterday["Low"]), 4),
                "close": round(float(today["Close"]), 4),
                "today_range": round(float(today_range), 4),
                "atr10": round(float(atr10), 4),
                "above_ema": above_ema,
                "compression_score": round(float(compression_score), 4),
                "market_ok": market_ok,
                "ema21_slope": round(float(ema21_slope), 6),
            }
        )

    print(f"Found {len(all_inside_days)} inside-day candidates.")

    # ---- SCORE + DIRECTION ----
    for sig in all_inside_days:
        base = WEIGHT_COMPRESSION * sig["compression_score"]
        if sig["above_ema"]:
            base += WEIGHT_TREND * 1.0
        if not sig["market_ok"]:
            base *= 0.7

        stock_bias = 1 if sig["above_ema"] else -1

        slope = sig["ema21_slope"]
        if slope > 0:
            slope_bias = 1
        elif slope < 0:
            slope_bias = -1
        else:
            slope_bias = 0

        if spy_ctx is not None:
            spy_bias = 1 if spy_ctx["above_ema21"] else -1
            combined = 0.5 * spy_bias + 0.3 * stock_bias + 0.2 * slope_bias
        else:
            combined = 0.6 * stock_bias + 0.4 * slope_bias

        direction = "CALL" if combined >= 0 else "PUT"

        sig["direction"] = direction
        sig["score"] = round(base, 4)

        # Sector filter decision (Mode 1) — only meaningful for single stocks, not SPY/QQQ/IWM/DIA
        if USE_SECTOR_FILTER and sig["ticker"] not in MAJOR_ETFS:
            ok, sec_etf = sector_ok_for_trade(sig["ticker"], direction, ticker_to_sector, sector_ctx)
            sig["sector_ok"] = bool(ok)
            sig["sector_etf"] = sec_etf
        else:
            sig["sector_ok"] = True
            sig["sector_etf"] = "NA"

    by_ticker = {sig["ticker"]: sig for sig in all_inside_days}

    # ---- SPREADS + TIER A ----
    tight_spread_inside_days = []
    for i, sig in enumerate(all_inside_days):
        if i >= CHECK_OPTIONS_FOR_MAX:
            break

        # Apply sector filter here so Tier A becomes "sector-approved"
        if USE_SECTOR_FILTER and not sig.get("sector_ok", True):
            continue

        best_spread = best_option_spread_for_ticker(sig["ticker"], sig["close"])
        if best_spread is not None:
            base_sig = by_ticker[sig["ticker"]].copy()
            base_sig["best_spread"] = round(float(best_spread), 4)
            base_sig["score"] = round(base_sig["score"] + WEIGHT_SPREAD, 4)
            tight_spread_inside_days.append(base_sig)

    all_inside_days = sorted(all_inside_days, key=lambda x: x["score"], reverse=True)
    tight_spread_inside_days = sorted(tight_spread_inside_days, key=lambda x: x["score"], reverse=True)

    return all_inside_days, tight_spread_inside_days


if __name__ == "__main__":
    all_setups, tight_setups = scan_inside_days()

    # --- TIER A ---
    if tight_setups:
        print("\nTIER A: inside-day setups WITH tight options + sector filter (highest priority)")
        print("ticker | score | dir   | date       | prev_high | prev_low | close   | ATR10 | best_spread | ema21_slope | sector_etf")
        for r in tight_setups:
            print(
                f"{r['ticker']:5s} | {r['score']:>5} | {r['direction']:<5} | {r['date']} | "
                f"{r['prev_high']:>8} | {r['prev_low']:>8} | "
                f"{r['close']:>7} | {r['atr10']:>5} | {r.get('best_spread', 'NA'):>11} | "
                f"{r['ema21_slope']:>10} | {r.get('sector_etf','NA'):>9}"
            )
    else:
        print("\nTIER A: (none with tight spreads in checked subset after sector filter)")

    # --- TIER B / full list ---
    print("\nTIER B: all other inside-day setups (ranked)")
    print("ticker | score | dir   | date       | prev_high | prev_low | close   | range  | ATR10 | above_ema | mkt_ok | ema21_slope | sector_ok | sector_etf")
    for r in all_setups:
        print(
            f"{r['ticker']:5s} | {r['score']:>5} | {r['direction']:<5} | {r['date']} | "
            f"{r['prev_high']:>8} | {r['prev_low']:>8} | "
            f"{r['close']:>7} | {r['today_range']:>6} | {r['atr10']:>5} | "
            f"{str(r['above_ema']):>9} | {str(r['market_ok']):>6} | {r['ema21_slope']:>10} | "
            f"{str(r.get('sector_ok', True)):>9} | {r.get('sector_etf','NA'):>9}"
        )

    print("\nPlan:")
    print("- Trade TIER A first — structure + liquidity + sector-aligned.")
    print("- Trigger = break above prev_high (tiny buffer); below prev_low invalidates.")
    print("- Direction = CALL/PUT from combined SPY + stock + EMA21-slope signal.")
    print("- Sector filter (Mode 1): CALL requires sector ETF > EMA21, PUT requires sector ETF < EMA21.")
