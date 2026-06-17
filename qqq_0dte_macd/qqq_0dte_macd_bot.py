"""
QQQ 0DTE MACD bot — PAPER-FORWARD test harness.

Strategy (validated in backtest_macd_0dte.py over 2023-08..2026-06, 2.8yr):
  - Entry: first 1-min MACD(12/26/9) crossover at/after 8:45 CST (9:45 ET) that
    agrees with session VWAP (longs above VWAP, shorts below). One trade per day.
  - Instrument: ATM 0DTE QQQ option (call for bull cross, put for bear).
  - Exit (scale-out / "balanced"): stop = -50% of premium (1R). Sell HALF at +100%
    (1:2), move stop to breakeven, run the rest to +150% (1:3). Hard flat 14:45 CST.
  - Backtest with VWAP + scale-out: ~42% WR, expR +0.22, alive in 2026 (PF ~1.4).

SAFETY: hard-guarded to the PAPER port (4002). Pass --allow-live to override (don't).
All option PnL was BS-simulated in the backtest — this paper run is to validate the
modeled edge against REAL fills before any real capital is considered.

Run (IB Gateway/TWS logged into PAPER account):
  python qqq_0dte_macd_bot.py            # live paper loop until EOD
  python qqq_0dte_macd_bot.py --dry-run  # full logic, logs intended orders, places none
  python qqq_0dte_macd_bot.py --status   # print state, no connection

Intended to be launched once per trading morning (~8:40 CST) by Task Scheduler.
"""

import argparse
import logging
import sys
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

if sys.version_info >= (3, 10):
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, Option

import ib_core
# single source of truth for signal + option pricing — imported from the backtest
from backtest_macd_0dte import (add_macd, ENTRY_AFTER_ET, EOD_CLOSE_ET, MACD_SLOW,
                                MACD_SIGNAL, bs_price, yearfrac_to_expiry)

# =========================================================  CONFIG
IB_PORT = 4002            # PAPER. 4001 = live (guarded below).
IB_CLIENT_ID = 8          # distinct from inside_day(2/5), scalper(1), dual_mom(3), review(4)
SYMBOL = "QQQ"

# Premium source for strike-pick + bracket management:
#   "model" — Black-Scholes from live QQQ spot (no OPRA needed). Use until subscribed.
#   "live"  — real option bid/ask via reqMktData (requires OPRA subscription).
# Flip to "live" (or pass --live) once the OPRA market-data subscription is active.
MANAGEMENT_MODE = "model"
IV_FALLBACK = 0.20        # used if VXN fetch fails (model mode IV proxy)

STOP_PCT = 0.50           # 1R = 50% of entry premium
SCALE_AT = 2.0            # sell half at +100% (1:2)
RUNNER_AT = 3.0           # run remainder to +150% (1:3)
OTM_PCT = 0.0             # 0 = ATM (the validated config). e.g. 0.004 = ~0.4% OTM
                          # (cheaper contract, better modeled expectancy but proportionally
                          # wider real spreads — paper-test before trusting). Override: --otm-pct
RISK_PCT = 0.01           # ~1% of NLV at risk per trade (the 50% stop loss)
MAX_CONTRACTS = 20        # cap (keeps paper fills realistic; raise consciously)
MAX_SPREAD_PCT = 0.12     # skip entry if ATM 0DTE spread wider than this
POLL_SECONDS = 20

# --- connection / data-outage hardening ---
CONNECT_RETRIES = 3       # IB connect attempts at startup
CONNECT_BACKOFF = 10      # seconds between connect attempts
ALERT_THROTTLE_SEC = 300  # min seconds between repeat alerts of the same IB error category
OUTAGE_ALERT_MIN = 3      # alert if no usable bars for this many minutes during market hours

ET = ZoneInfo("America/New_York")
MARKET_OPEN_ET = dtime(9, 30)
# Only need a few bars for the EMA recursion + shift(1). Timing is controlled by the
# 9:45 ET eligibility filter in compute_signal — NOT by a bar count. A high floor here
# would skip early-window (9:45-10:10 ET) crossovers that the backtest takes. EMA with
# adjust=False at bar i is identical whether computed over 15 bars or the full session,
# so partial-session computation matches the backtest exactly.
MIN_BARS = 5

STATE_FILE = "qqq_0dte_state.json"
LOG_FILE = "qqq_0dte_macd.log"
JOURNAL_FILE = "qqq_0dte_journal.csv"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1494853248547553300/nlPc2xVTnA-WTEMbfr0TroLJMbmgJ8VYjq_7W9J7rOqEfuHAVSnqchAIAkhIy9QkOqLr"

DEFAULT_STATE = {"date": None, "traded_today": False, "position": None,
                 "cumulative_pnl": 0.0, "trades_taken": 0}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("qqq0dte")


def now_et() -> datetime:
    return datetime.now(ET)


def alert(msg: str, urgent: bool = False):
    logger.info(f"DISCORD: {msg}")
    ib_core.send_discord(msg, DISCORD_WEBHOOK_URL, logger, tag="QQQ-0DTE", urgent=urgent)


# IB connectivity / data-farm error codes worth shouting about (benign "OK" and
# informational codes like 2104/2106/2107/2158 are intentionally NOT here).
_ALERT_CODES = {1100, 1101, 1102, 2103, 2105, 2110}
_last_err_alert: dict[str, datetime] = {}


def _on_ib_error(reqId, errorCode, errorString, contract=None, *extra):
    """Turn the silent error-event stream into loud, throttled Discord alerts.
    Catches the exact failure modes seen on 2026-06-17: the 'different IP address'
    session conflict and the 'HMDS query returned no data' blackout that followed."""
    msg = errorString or ""
    if "different IP address" in msg:
        cat = "session IP conflict"
    elif errorCode == 162 and "no data" in msg.lower():
        cat = "historical-data outage"
    elif errorCode in _ALERT_CODES:
        cat = f"connectivity (code {errorCode})"
    else:
        return  # benign / informational — ignore
    now = now_et()
    last = _last_err_alert.get(cat)
    if last is None or (now - last).total_seconds() >= ALERT_THROTTLE_SEC:
        _last_err_alert[cat] = now
        alert(f"IB {cat}: {msg.strip()} (code {errorCode})", urgent=True)


def connect_with_retry() -> IB:
    """Connect to IB Gateway, retrying a few times before giving up loudly."""
    import time as _time
    last_exc = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            return ib_core.connect_ib(IB_PORT, IB_CLIENT_ID, logger)
        except Exception as e:
            last_exc = e
            logger.warning(f"IB connect attempt {attempt}/{CONNECT_RETRIES} failed: {e}")
            if attempt < CONNECT_RETRIES:
                _time.sleep(CONNECT_BACKOFF)
    alert(f"CONNECT FAILED after {CONNECT_RETRIES} attempts: {last_exc}", urgent=True)
    raise last_exc


def fetch_current_iv() -> float:
    """ATM IV proxy from latest VXN close (model mode). Falls back to IV_FALLBACK."""
    try:
        import yfinance as yf
        v = yf.Ticker("^VXN").history(period="5d", interval="1d", auto_adjust=False)
        return float(v["Close"].iloc[-1]) / 100.0
    except Exception as e:
        logger.warning(f"VXN fetch failed ({e}); using IV={IV_FALLBACK}")
        return IV_FALLBACK


def model_premium(strike: float, right: str, spot: float, iv: float) -> float:
    """Black-Scholes 0DTE mark from current QQQ spot (matches the backtest pricing)."""
    T = yearfrac_to_expiry(pd.Timestamp(now_et()))
    return bs_price(spot, float(strike), T, iv, right)


def current_premium(ib: IB, pos: dict, spot: float, iv: float):
    """Current option mark — real bid/ask mid in 'live' mode, BS model in 'model' mode."""
    if MANAGEMENT_MODE == "live":
        contract = Option(pos["symbol"], pos["expiry"], pos["strike"], pos["right"], "SMART")
        ib.qualifyContracts(contract)
        _, _, mid = ib_core.option_mid(ib, contract)
        return mid
    return model_premium(pos["strike"], pos["right"], spot, iv)


# =========================================================  DATA / SIGNAL
def fetch_closed_bars(ib: IB, qqq) -> pd.DataFrame | None:
    """Today's RTH 1-min bars, EXCLUDING the in-progress current minute."""
    bars = ib.reqHistoricalData(qqq, endDateTime="", durationStr="1 D",
                                barSizeSetting="1 min", whatToShow="TRADES",
                                useRTH=True, formatDate=1)
    if not bars:
        return None
    df = pd.DataFrame(
        [(b.date, b.open, b.high, b.low, b.close, b.volume) for b in bars],
        columns=["date", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = (df["date"].dt.tz_localize(ET) if df["date"].dt.tz is None
                  else df["date"].dt.tz_convert(ET))
    # drop the still-forming bar (its minute == now)
    cutoff = now_et().replace(second=0, microsecond=0)
    df = df[df["date"] < cutoff].reset_index(drop=True)
    return df


def compute_signal(df: pd.DataFrame) -> dict | None:
    """First crossover at/after 9:45 ET that agrees with session VWAP. Mirrors backtest."""
    d = add_macd(df).reset_index(drop=True)
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    d["vwap"] = (tp * d["volume"]).cumsum() / d["volume"].cumsum().replace(0, np.nan)
    cross_up = (d["macd"] > d["signal"]) & (d["macd"].shift(1) <= d["signal"].shift(1))
    cross_dn = (d["macd"] < d["signal"]) & (d["macd"].shift(1) >= d["signal"].shift(1))
    eligible = d["date"].dt.time >= ENTRY_AFTER_ET
    for cand in d.index[(cross_up | cross_dn) & eligible]:
        is_long = bool(cross_up.iloc[cand])
        px, vw = d["close"].iloc[cand], d["vwap"].iloc[cand]
        if is_long and px < vw:        # VWAP filter
            continue
        if not is_long and px > vw:
            continue
        return {"idx": int(cand), "is_last": cand == len(d) - 1,
                "direction": "long" if is_long else "short",
                "time": d["date"].iloc[cand], "spot": float(d["close"].iloc[cand]),
                "cross": "up" if is_long else "dn",
                "macd": float(d["macd"].iloc[cand]), "signal": float(d["signal"].iloc[cand]),
                "vwap": float(vw)}
    return None


# =========================================================  OPTION SELECTION
def select_0dte_option(ib: IB, direction: str, spot: float, iv: float) -> dict | None:
    """0DTE QQQ option in the signal direction. ATM when OTM_PCT==0, else shifted
    OTM_PCT out-of-the-money (matches the backtest's otm_pct). Tries the anchor
    strike, then +/-1. 'live' mode uses real bid/ask + spread filter; 'model' prices via BS."""
    expiry = now_et().strftime("%Y%m%d")
    right = "C" if direction == "long" else "P"
    if OTM_PCT and right == "C":
        base = round(spot * (1 + OTM_PCT))
    elif OTM_PCT and right == "P":
        base = round(spot * (1 - OTM_PCT))
    else:
        base = round(spot)
    for strike in (base, base + 1, base - 1):
        opt = Option(SYMBOL, expiry, float(strike), right, "SMART")
        try:
            if not ib.qualifyContracts(opt):
                continue
        except Exception:
            continue
        if MANAGEMENT_MODE == "live":
            bid, ask, mid = ib_core.option_mid(ib, opt)
            if mid is None:
                continue
            spread = (ask - bid) / mid if mid > 0 else 1.0
            if spread > MAX_SPREAD_PCT:
                logger.info(f"  {SYMBOL} {right}{strike} {expiry}: spread {spread:.1%} > {MAX_SPREAD_PCT:.0%}")
                continue
            return {"contract": opt, "strike": float(strike), "right": right,
                    "expiry": expiry, "bid": bid, "ask": ask, "mid": mid, "spread": spread}
        # model mode: no option quotes; mark via BS from spot
        mid = model_premium(strike, right, spot, iv)
        if mid <= 0.05:
            continue
        return {"contract": opt, "strike": float(strike), "right": right,
                "expiry": expiry, "bid": None, "ask": None, "mid": mid, "spread": None}
    return None


# =========================================================  ENTRY / MANAGEMENT
def _open_position(state: dict, opt: dict, sig: dict, fill: float, qty: int):
    """Record an open position + bracket marks and persist. Shared by the live and
    dry-run paths so paper runs manage/close exactly as a live run would."""
    state["position"] = {
        "symbol": SYMBOL, "expiry": opt["expiry"], "strike": opt["strike"], "right": opt["right"],
        "qty_total": qty, "qty_open": qty, "entry_price": fill,
        "entry_time": now_et().isoformat(), "direction": sig["direction"],
        "spot_at_entry": sig["spot"],
        "stop_mark": round(fill * (1 - STOP_PCT), 2),
        "first_target": round(fill * (1 + STOP_PCT * SCALE_AT), 2),
        "runner_target": round(fill * (1 + STOP_PCT * RUNNER_AT), 2),
        "scaled": False, "stop_at_be": False, "realized": 0.0,
    }
    state["trades_taken"] = state.get("trades_taken", 0) + 1
    ib_core.save_state(STATE_FILE, state)


def enter(ib: IB, state: dict, sig: dict, spot: float, iv: float, dry_run: bool):
    # audit trail: why this direction was chosen (cross type + price vs VWAP)
    logger.info(f"SIGNAL {sig['direction']} (MACD cross {sig['cross']}) @ "
                f"{sig['time'].strftime('%H:%M')} ET | macd={sig['macd']:+.4f} "
                f"signal={sig['signal']:+.4f} | close={sig['spot']:.2f} vwap={sig['vwap']:.2f} "
                f"({'above' if sig['spot'] >= sig['vwap'] else 'below'} VWAP)")
    opt = select_0dte_option(ib, sig["direction"], spot, iv)
    if not opt:
        alert(f"Signal fired ({sig['direction']} @ {spot:.2f}) but no tradeable ATM 0DTE "
              f"(spread/qualify). Skipping — one-shot for today.", urgent=True)
        state["traded_today"] = True
        ib_core.save_state(STATE_FILE, state)
        return

    nlv = ib_core.get_account_balance(ib) or 0.0
    cost_ct = opt["mid"] * 100
    # position premium sized so a 50% stop ≈ RISK_PCT of NLV
    target_premium = (nlv * RISK_PCT) / STOP_PCT if nlv > 0 else cost_ct
    qty = max(1, min(MAX_CONTRACTS, int(target_premium / cost_ct))) if cost_ct > 0 else 1

    spread_str = f"{opt['spread']:.1%}" if opt["spread"] is not None else f"model(iv {iv:.0%})"
    logger.info(f"ENTRY {sig['direction']} {SYMBOL} {opt['right']}{opt['strike']} {opt['expiry']} "
                f"mid={opt['mid']:.2f} qty={qty} (NLV {nlv:.0f}, {spread_str})")
    state["traded_today"] = True
    if dry_run:
        fill = opt["mid"]  # simulated fill at modeled/quoted mid
        alert(f"[DRY] would BUY {qty}x {SYMBOL} {opt['right']}{opt['strike']} 0DTE @~{fill:.2f}")
        _open_position(state, opt, sig, fill, qty)
        return

    res = ib_core.place_buy(ib, opt["contract"], qty, opt["mid"], logger)
    if not res["filled"]:
        ib_core.save_state(STATE_FILE, state)
        alert(f"FAILED to buy {SYMBOL} {opt['right']}{opt['strike']} 0DTE — no fill.", urgent=True)
        return
    fill = res["fill_price"]
    _open_position(state, opt, sig, fill, qty)
    p = state["position"]
    alert(f"OPENED {qty}x {SYMBOL} {p['right']}{p['strike']} 0DTE @ {fill:.2f} "
          f"({sig['direction']}, spot {sig['spot']:.2f}) | stop {p['stop_mark']} "
          f"scale {p['first_target']} runner {p['runner_target']}")


def _sell(ib: IB, pos: dict, qty: int, mid_hint: float, dry_run: bool) -> float | None:
    """Sell `qty` contracts to close; return fill price or None."""
    contract = Option(pos["symbol"], pos["expiry"], pos["strike"], pos["right"], "SMART")
    ib.qualifyContracts(contract)
    if dry_run:
        return mid_hint
    res = ib_core.place_sell(ib, contract, qty, mid_hint, logger)
    return res["fill_price"] if res["filled"] else None


def manage(ib: IB, state: dict, spot: float, iv: float, dry_run: bool):
    pos = state["position"]
    mid = current_premium(ib, pos, spot, iv)
    if mid is None:
        return  # no quote this poll
    cur_stop = pos["entry_price"] if pos["stop_at_be"] else pos["stop_mark"]

    # 1) stop (worst-case checked first)
    if mid <= cur_stop:
        fill = _sell(ib, pos, pos["qty_open"], mid, dry_run)
        _book_close(state, fill if fill is not None else mid, "be_stop" if pos["stop_at_be"] else "stop")
        return
    # 2) scale half at first target -> stop to breakeven
    if (not pos["scaled"]) and pos["qty_open"] >= 2 and mid >= pos["first_target"]:
        half = pos["qty_open"] // 2
        fill = _sell(ib, pos, half, mid, dry_run)
        if fill is not None:
            pos["realized"] += (fill - pos["entry_price"]) * 100 * half
            pos["qty_open"] -= half
            pos["scaled"] = True
            pos["stop_at_be"] = True
            ib_core.save_state(STATE_FILE, state)
            alert(f"SCALED {half}x {pos['symbol']} {pos['right']}{pos['strike']} @ {fill:.2f} "
                  f"(+{STOP_PCT*SCALE_AT:.0%}); stop->breakeven, running {pos['qty_open']}x to "
                  f"{pos['runner_target']}")
    # 3) runner target -> close remainder
    if mid >= pos["runner_target"]:
        fill = _sell(ib, pos, pos["qty_open"], mid, dry_run)
        _book_close(state, fill if fill is not None else mid, "runner")


def _book_close(state: dict, fill: float, reason: str):
    pos = state["position"]
    leg = (fill - pos["entry_price"]) * 100 * pos["qty_open"]
    total = round(pos["realized"] + leg, 2)
    state["cumulative_pnl"] = round(state.get("cumulative_pnl", 0.0) + total, 2)
    ib_core.append_journal(JOURNAL_FILE, {
        "symbol": pos["symbol"], "right": pos["right"], "strike": pos["strike"],
        "expiry": pos["expiry"], "direction": pos["direction"],
        "entry_time": pos["entry_time"], "exit_time": now_et().isoformat(),
        "entry_price": pos["entry_price"], "exit_price": fill,
        "qty_total": pos["qty_total"], "exit_reason": reason,
        "spot_at_entry": pos["spot_at_entry"], "pnl": total,
    })
    state["position"] = None
    ib_core.save_state(STATE_FILE, state)
    alert(f"CLOSED {pos['symbol']} {pos['right']}{pos['strike']} ({reason}) @ {fill:.2f} "
          f"for ${total:+.2f}. Cum P&L ${state['cumulative_pnl']:+.2f}")


def close_all_eod(ib: IB, state: dict, spot: float, iv: float, dry_run: bool):
    pos = state["position"]
    mid = current_premium(ib, pos, spot, iv)
    fill = _sell(ib, pos, pos["qty_open"], mid if mid else 0.05, dry_run)
    _book_close(state, fill if fill is not None else (mid or 0.05), "eod")


# =========================================================  MAIN LOOP
def run(dry_run: bool = False):
    state = ib_core.load_state(STATE_FILE, DEFAULT_STATE)
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:  # daily reset (keep cumulative_pnl + position if somehow open)
        state.update({"date": today, "traded_today": False})
        if state.get("position") is None:
            state["position"] = None
        ib_core.save_state(STATE_FILE, state)

    ib = connect_with_retry()
    ib.errorEvent += _on_ib_error   # loud, throttled alerts on IP-conflict / data outages
    qqq = ib.qualifyContracts(Stock(SYMBOL, "SMART", "USD"))[0]
    iv = fetch_current_iv() if MANAGEMENT_MODE == "model" else 0.0
    logger.info("=" * 60)
    logger.info(f"QQQ 0DTE MACD BOT {'[DRY-RUN] ' if dry_run else ''}— {today} | mode={MANAGEMENT_MODE}"
                f"{f' iv={iv:.0%}' if MANAGEMENT_MODE == 'model' else ''} | "
                f"strike={'ATM' if not OTM_PCT else f'{OTM_PCT:.1%} OTM'} | "
                f"traded_today={state['traded_today']} position={bool(state['position'])} | "
                f"cum P&L ${state.get('cumulative_pnl', 0):+.2f}")
    logger.info("=" * 60)

    outage_since = None       # tracks a market-hours data blackout
    outage_alerted = False
    try:
        while True:
            now = now_et()
            if now.time() >= EOD_CLOSE_ET:
                if state["position"]:
                    logger.info("EOD reached with open position — flattening.")
                    df = fetch_closed_bars(ib, qqq)
                    eod_spot = (float(df["close"].iloc[-1]) if df is not None and len(df)
                                else state["position"]["spot_at_entry"])
                    close_all_eod(ib, state, eod_spot, iv, dry_run)
                logger.info("EOD close time reached. Done for the day.")
                break
            if now.time() < MARKET_OPEN_ET:
                ib.sleep(POLL_SECONDS)
                continue

            try:
                df = fetch_closed_bars(ib, qqq)
                if df is None or len(df) < MIN_BARS:
                    # data blackout in market hours — alert only when it can actually
                    # cost us: no trade taken yet, or a live position we're now blind to
                    if state["position"] or not state["traded_today"]:
                        if outage_since is None:
                            outage_since = now
                        elif (not outage_alerted and
                              (now - outage_since).total_seconds() >= OUTAGE_ALERT_MIN * 60):
                            alert(f"DATA OUTAGE: no usable {SYMBOL} bars for "
                                  f"{OUTAGE_ALERT_MIN}+ min in market hours — a signal or "
                                  f"exit could be missed. Check IB Gateway feed.", urgent=True)
                            outage_alerted = True
                    ib.sleep(POLL_SECONDS)
                    continue
                if outage_alerted:
                    alert(f"DATA RESTORED: {SYMBOL} bars flowing again.")
                outage_since, outage_alerted = None, False
                spot = float(df["close"].iloc[-1])

                if state["position"]:
                    manage(ib, state, spot, iv, dry_run)
                elif not state["traded_today"]:
                    sig = compute_signal(df)
                    if sig and sig["is_last"]:
                        enter(ib, state, sig, spot, iv, dry_run)
                    elif sig and not sig["is_last"]:
                        # first qualifying cross already happened before we were watching
                        # (late start / restart) — don't chase; honor one-shot.
                        alert(f"Missed first signal ({sig['direction']} at "
                              f"{sig['time'].strftime('%H:%M')} ET) — skipping, one trade/day.")
                        state["traded_today"] = True
                        ib_core.save_state(STATE_FILE, state)
            except Exception as e:
                logger.exception("Loop iteration error (continuing)")
                alert(f"loop error: {e}", urgent=True)

            ib.sleep(POLL_SECONDS)
    finally:
        ib.disconnect()
        logger.info("Disconnected from IB.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="run logic, place no orders")
    ap.add_argument("--status", action="store_true", help="print state and exit")
    ap.add_argument("--allow-live", action="store_true", help="permit non-paper port (DON'T)")
    ap.add_argument("--model", action="store_true", help="force BS-model premium mode (no OPRA)")
    ap.add_argument("--live", action="store_true", help="force real option quotes (needs OPRA)")
    ap.add_argument("--otm-pct", type=float, default=None, help="OTM strike offset (e.g. 0.004); 0=ATM")
    args = ap.parse_args()

    global MANAGEMENT_MODE, OTM_PCT
    if args.model:
        MANAGEMENT_MODE = "model"
    elif args.live:
        MANAGEMENT_MODE = "live"
    if args.otm_pct is not None:
        OTM_PCT = args.otm_pct

    if args.status:
        import json
        print(json.dumps(ib_core.load_state(STATE_FILE, DEFAULT_STATE), indent=2, default=str))
        return

    if IB_PORT != 4002 and not args.allow_live:
        logger.error(f"IB_PORT={IB_PORT} is not the paper port (4002). Refusing to start "
                     f"without --allow-live. This strategy is NOT validated for real capital.")
        sys.exit(2)

    try:
        run(dry_run=args.dry_run)
    except Exception as e:
        logger.exception("Bot crashed")
        alert(f"CRASHED: {e}", urgent=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
