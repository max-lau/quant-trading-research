# Milestone 0 — Foundations

## Order types
- Market order: immediate fill, pay the spread, price uncertain
- Limit order: guaranteed price, fill uncertain
- Stop order: triggers a market order once a price level is hit; can gap in fast markets

## Bid-ask spread
Gap between best bid and best ask. Crossing it (market order) is a real, silent cost.
This is why every backtest must be re-run with realistic transaction costs (Milestone 3).

## Vectorization
Loops over DataFrames are a red flag in interviews and production code.
pandas/numpy operations (shift, rolling, log, diff) are implemented in C and process
whole arrays at once — see m0_vectorization_check.py for a measured ~17x/row speedup.
Exception: genuinely stateful, path-dependent logic (e.g. a signal that depends on the
previous bar's position) sometimes has to loop — but even then, isolate that loop and
vectorize everything around it.
