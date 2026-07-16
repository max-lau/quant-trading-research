"""
Milestone 1 — synthetic OHLCV generator.

Why simulate at all, when the goal is real market data?
Two honest reasons:
  1. It lets us develop and unit-test the *pipeline* (cleaning, gap detection,
     storage) without depending on network access or rate limits.
  2. A simulator with a KNOWN ground truth (known regimes, known drift) is the
     only way to verify your feature/label code is correct — you can check
     "does my momentum feature correctly detect the regime I injected?"
     before ever trusting it on real, noisy data.

This is NOT a substitute for real data in Milestones 3+. See fetch_real_data.py
for the live version to run on your own machine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _gbm_regime_path(rng: np.random.Generator, n_days: int, p0: float,
                      vol_lo: float, vol_hi: float) -> pd.DataFrame:
    """One ticker's price path: regime-switching geometric Brownian motion."""
    price = p0
    regime = 0
    drifts = [0.0006, 0.0000, -0.0010]     # calm-up, choppy, stressed
    rows = []
    for _ in range(n_days):
        if rng.random() < 0.03:
            regime = rng.integers(0, 3)
        vol = vol_lo + (vol_hi - vol_lo) * (regime / 2) + 0.003 * abs(rng.normal())
        drift = drifts[regime]
        o = price
        c = o * np.exp(drift + vol * rng.normal())
        h = max(o, c) * (1 + abs(rng.normal()) * vol * 0.5)
        l = min(o, c) * (1 - abs(rng.normal()) * vol * 0.5)
        v = int(1e6 * (0.6 + regime * 0.5 + abs(rng.normal())))
        price = c
        rows.append((o, h, l, c, v, regime))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume", "regime"])


def simulate_universe(tickers: dict[str, dict], start: str, end: str,
                       inject_delisting: bool = True) -> pd.DataFrame:
    """
    Build a synthetic multi-ticker OHLCV panel.

    tickers: {name: {"seed": int, "p0": float, "vol_lo": float, "vol_hi": float}}
    inject_delisting: if True, one ticker "goes bankrupt" partway through and
        stops trading — deliberately, so you can practice detecting and
        handling survivorship bias instead of only reading about it.
    """
    dates = pd.bdate_range(start, end)
    frames = []
    delisted_ticker = list(tickers.keys())[-1] if inject_delisting and len(tickers) > 1 else None

    for name, cfg in tickers.items():
        rng = np.random.default_rng(cfg["seed"])
        df = _gbm_regime_path(rng, len(dates), cfg["p0"], cfg["vol_lo"], cfg["vol_hi"])
        df.insert(0, "date", dates)
        df.insert(0, "ticker", name)

        if name == delisted_ticker:
            # simulate a delisting event ~70% through the sample: price craters, then vanishes
            cut = int(len(df) * 0.7)
            crash_len = 15
            df.loc[cut:cut + crash_len, "close"] *= np.linspace(1, 0.05, crash_len + 1)
            for col in ("open", "high", "low"):
                df.loc[cut:cut + crash_len, col] = df.loc[cut:cut + crash_len, "close"]
            df = df.iloc[:cut + crash_len + 1]  # ticker stops trading — this is the "hole" naive datasets erase

        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["close"] = panel["close"].round(2)
    panel["open"] = panel["open"].round(2)
    panel["high"] = panel["high"].round(2)
    panel["low"] = panel["low"].round(2)
    return panel


if __name__ == "__main__":
    universe = {
        "QNTX": {"seed": 11, "p0": 184, "vol_lo": .010, "vol_hi": .032},
        "AERO": {"seed": 47, "p0": 62, "vol_lo": .014, "vol_hi": .040},
        "HELIX": {"seed": 83, "p0": 311, "vol_lo": .008, "vol_hi": .026},
        "FLUX": {"seed": 29, "p0": 24, "vol_lo": .018, "vol_hi": .055},
        "OMEGA_CORP": {"seed": 5, "p0": 45, "vol_lo": .012, "vol_hi": .038},  # this one will "delist"
    }
    panel = simulate_universe(universe, "2020-01-01", "2025-12-31")
    panel.to_csv("data/simulated_ohlcv.csv", index=False)
    print(panel.groupby("ticker")["date"].agg(["min", "max", "count"]))
