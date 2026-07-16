"""
Quantify survivorship bias: what happens if you build your universe from
today's surviving tickers and apply it retroactively (the classic naive
mistake), vs. correctly including OMEGA_CORP for the period it was alive.
"""
import pandas as pd
import numpy as np

panel = pd.read_parquet("data/simulated_ohlcv.parquet")
panel = panel.sort_values(["ticker", "date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()

# Correct: equal-weight portfolio using every ticker that was actually
# tradeable on each date (OMEGA_CORP included until it delists).
correct_daily = panel.groupby("date")["ret"].mean()
correct_total_return = (1 + correct_daily).prod() - 1

# Naive / survivorship-biased: only use tickers that are STILL trading today,
# applied across the whole history -- silently drops OMEGA_CORP entirely,
# even from the years it was a perfectly normal, tradeable stock.
survivors = panel.groupby("ticker")["date"].max()
survivors = survivors[survivors == survivors.max()].index.tolist()
naive_daily = panel[panel["ticker"].isin(survivors)].groupby("date")["ret"].mean()
naive_total_return = (1 + naive_daily).prod() - 1

print(f"Survivors used in naive universe: {survivors}")
print(f"\nCorrect (point-in-time) equal-weight total return: {correct_total_return:+.1%}")
print(f"Naive (survivorship-biased) total return:           {naive_total_return:+.1%}")
print(f"\nBias inflation: {(naive_total_return - correct_total_return)*100:+.1f} percentage points")
print("\n-> The naive backtest looks better ONLY because it erased the one company that failed.")
