# Inside-Day Scanner (S&P 500 + Major ETFs) — Tight-Options Tier A

This script scans S&P 500 constituents (plus SPY/QQQ/IWM/DIA) for **inside-day** setups with **range compression**, applies a **trend filter**, optionally **penalizes setups when SPY is below its EMA21**, then checks the **nearest expiration option chain** and promotes only names with **tight bid/ask spreads** into **Tier A**.

It prints a **Tier A list (score >= 0.55)** with direction, scores, and key levels.

---

## What it finds

A ticker becomes a candidate if:

1. **Inside day**
   - Today’s **High <= yesterday’s High**
   - Today’s **Low >= yesterday’s Low**

2. **Range compression vs ATR**
   - `today_range = High - Low`
   - `ATR10 = 10-day average true range`
   - Passes if:  
     `today_range <= (RANGE_COMPRESSION_PCT * ATR10)`  
     Default `RANGE_COMPRESSION_PCT = 0.7`

3. **Trend context (EMA21)**
   - Computes **EMA21** on daily closes
   - `above_ema = Close > EMA21` when `USE_TREND_FILTER = True`

4. **Market context (SPY EMA21)**
   - SPY is used as a **filter-only penalty** (does **not** affect direction)
   - If SPY close is **below** its EMA21, the setup score is multiplied by `0.7`

5. **Options liquidity filter (tight spreads)**
   - For the highest scored inside-day candidates (default 120 tickers), it checks the **nearest expiration** options chain
   - Looks for strikes within **±2%** of spot (`ATM_MONEYNES_PCT = 0.02`)
   - Accepts a contract if either:
     - Absolute spread <= `MAX_ABS_SPREAD` (default **$0.15**), OR
     - Relative spread <= `MAX_REL_SPREAD` (default **1.5% of mid**)
   - If a ticker passes, it gets an added score bump (`WEIGHT_SPREAD`)

---

## Direction logic (CALL vs PUT)

Direction is based on the stock only:

- `stock_bias = +1` if price is above EMA21 else `-1`
- `ema21_slope_norm = (EMA21[t] - EMA21[t-5]) / ATR10` (normalized to avoid price-level bias)
- `combined = 0.6*stock_bias + 0.4*clamp(ema21_slope_norm, -1, 1)`
- `CALL` if `combined >= 0`, else `PUT`

**SPY does NOT decide CALL/PUT** — it only penalizes the score when SPY is bearish.

---

## Scoring

Default weights:

- Compression: `WEIGHT_COMPRESSION = 0.5`
- Trend: `WEIGHT_TREND = 0.3`
- Spread bonus: `WEIGHT_SPREAD = 0.2`

Score is computed as:

- `compression_score = 1 - (today_range / ATR10)` (floored at 0)
- `base = 0.5*compression_score`
- if `above_ema`: `base += 0.3`
- if SPY is below EMA21: `base *= 0.7`
- if tight options spread found: `score += 0.2`

Tier A prints only:
- **tight-spread tickers** with `score >= 0.55`

---

## Output format

Example header:



