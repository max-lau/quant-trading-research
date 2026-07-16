"""
Milestone 3, part 2 -- two classic, economically-motivated strategies.

TIME-SERIES MOMENTUM: go long an asset that has recently risen, short one
that has recently fallen. Economic story: underreaction to news, herding,
and slow information diffusion mean trends persist longer than a random
walk would predict. This is one of the most robust, widely-documented
effects in finance (Moskowitz, Ooi & Pedersen 2012 is the canonical paper).

MEAN REVERSION: bet that a large short-term deviation from a recent average
will partially reverse. Economic story: short-term overreaction, liquidity
provision (someone has to absorb a panic sell order, and gets paid a
premium for doing so), and the bid-ask bounce we found in NVDA's raw
returns. Much shorter shelf-life than momentum -- works at short horizons,
decays fast, and is exactly the kind of effect transaction costs can erase
completely (see part 3).

Both signals here are DELIBERATELY simple (no parameter search, no
optimization) -- they're baselines Milestone 4's ML model needs to beat,
not a final product.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def momentum_signal(close: pd.Series, lookback: int = 60) -> pd.Series:
    """+1 if price is above its level `lookback` days ago, -1 if below."""
    past = close.shift(lookback)
    return np.sign(close / past - 1).fillna(0.0)


def mean_reversion_signal(close: pd.Series, lookback: int = 5, z_entry: float = 1.0) -> pd.Series:
    """
    -1 (short) if price is z_entry std devs ABOVE its rolling mean (bet on
    pullback), +1 (long) if z_entry std devs BELOW (bet on bounce), else flat.
    """
    roll_mean = close.rolling(lookback).mean()
    roll_std = close.rolling(lookback).std()
    z = (close - roll_mean) / roll_std
    position = pd.Series(0.0, index=close.index)
    position[z > z_entry] = -1.0
    position[z < -z_entry] = 1.0
    return position
