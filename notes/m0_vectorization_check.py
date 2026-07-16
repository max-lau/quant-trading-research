"""
Milestone 0 check: loop vs vectorized equivalence + speed.
Task: compute daily log returns and a 20-day rolling volatility
for a large synthetic price series, two ways.
"""
import numpy as np
import pandas as pd
import time

rng = np.random.default_rng(7)
n = 2_000_000  # ~ 8000 years of daily data, deliberately large to make the gap obvious
prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))

# --- Loop version (what NOT to do in a quant interview) ---
t0 = time.perf_counter()
log_rets_loop = [None]
for i in range(1, 5000):  # small slice only — full loop would take too long
    log_rets_loop.append(np.log(prices[i] / prices[i-1]))
t_loop = time.perf_counter() - t0

# --- Vectorized version ---
t0 = time.perf_counter()
log_rets_vec = np.log(prices / prices.shift(1))
roll_vol = log_rets_vec.rolling(20).std()
t_vec = time.perf_counter() - t0

print(f"Loop time for 5,000 rows:      {t_loop*1000:.2f} ms")
print(f"Vectorized time for {n:,} rows: {t_vec*1000:.2f} ms")
print(f"\nSanity check — first 5 vectorized log returns:\n{log_rets_vec.head()}")
print(f"\n20d rolling vol, rows 20-24:\n{roll_vol.iloc[20:25]}")
