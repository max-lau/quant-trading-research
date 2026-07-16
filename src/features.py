"""
Milestone 4, part 1 -- feature engineering.

A "feature" here is just a number computed FROM PAST DATA ONLY, for each day,
that we hope is informative about what happens next. The golden rule:

    A feature computed for date T must use ONLY data available up through
    date T's close. Never data from T+1 onward.

This sounds obvious, but it's easy to break by accident -- e.g. using
`.rolling(20).mean()` is fine (it looks backward), but centering a window
(`.rolling(20, center=True).mean()`) silently looks INTO THE FUTURE and is a
classic, hard-to-spot lookahead bug. Every function below is backward-looking
only, by construction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_features(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """
    Each column is one feature, computed for every date using only that
    date's close/volume and earlier. Returns a DataFrame aligned to `close`'s
    index -- early rows will be NaN until enough history exists (e.g. a
    60-day feature can't be computed until day 61).
    """
    feats = pd.DataFrame(index=close.index)

    # Momentum features: recent return over different lookback windows.
    # "how much has price moved over the last N days" -- captures trend.
    for lb in (5, 20, 60):
        feats[f"mom_{lb}d"] = close.pct_change(lb)

    # Volatility feature: how turbulent has the last 20 days been?
    # (we already know from Milestone 2 that volatility clusters -- this
    # lets the model condition on the current regime)
    log_ret = np.log(close / close.shift(1))
    feats["vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

    # RSI-style mean-reversion feature: is price stretched vs its own
    # recent average, in standardized units?
    roll_mean = close.rolling(10).mean()
    roll_std = close.rolling(10).std()
    feats["zscore_10d"] = (close - roll_mean) / roll_std

    # Volatility-of-volatility: is the vol regime itself unstable? (ratio of
    # short vs long vol, a simple regime-shift proxy)
    vol_5 = log_ret.rolling(5).std()
    vol_60 = log_ret.rolling(60).std()
    feats["vol_ratio_5_60"] = vol_5 / vol_60

    if volume is not None:
        # Volume surprise: today's volume vs its own recent normal level.
        feats["vol_surge"] = volume / volume.rolling(20).mean() - 1

    return feats


def build_label_fixed_horizon(close: pd.Series, horizon: int = 5) -> pd.Series:
    """
    THE SIMPLEST POSSIBLE LABEL: did price rise over the next `horizon` days?
    1 = yes, 0 = no. This uses FUTURE data by design -- that's fine, because
    it's the answer key, not an input feature. The critical rule is that this
    column must NEVER be used to compute a feature.

    Note the overlapping-labels problem this creates: the label for day T
    and the label for day T+1 both depend on prices in [T, T+horizon] and
    [T+1, T+horizon+1] -- these overlap almost entirely for horizon=5. This
    means consecutive rows are NOT independent samples, which is exactly why
    a naive shuffled K-fold cross-validation leaks information (a training
    row and a test row can share most of their underlying price data). We'll
    fix this properly with purged cross-validation in part 2.
    """
    fwd_ret = close.shift(-horizon) / close - 1
    label = (fwd_ret > 0).astype(float)
    label[fwd_ret.isna()] = np.nan  # can't label the last `horizon` days -- no future data exists yet
    return label


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_ohlcv.csv"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"

    panel = pd.read_csv(path, parse_dates=["date"])
    grp = panel.loc[panel["ticker"] == ticker].sort_values("date").set_index("date")

    feats = build_features(grp["close"], grp["volume"])
    label = build_label_fixed_horizon(grp["close"], horizon=5)

    dataset = feats.copy()
    dataset["label"] = label
    dataset = dataset.dropna()

    print(f"{ticker}: {len(dataset)} usable rows after dropping NaN warm-up/tail periods")
    print(f"\nFeature preview:\n{dataset.head()}")
    print(f"\nLabel balance: {dataset['label'].mean():.1%} of days are 'up' over the next 5 days")
    print("(if this is far from 50%, that's just the asset's overall drift over this period --")
    print(" not a sign the label is broken)")
