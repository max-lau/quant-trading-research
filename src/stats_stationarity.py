"""
Milestone 2, part 3 -- stationarity, and why "statistically significant"
is not the same as "real, tradeable edge".

PART A: Augmented Dickey-Fuller (ADF) test for stationarity.
  A stationary series has constant mean/variance over time -- prices are
  NOT stationary (they trend, wander, have no fixed mean), but RETURNS
  usually are. Almost every statistical/ML technique (correlation, most
  regressions, standard cross-validation assumptions) implicitly assumes
  some form of stationarity. Feeding a model raw, non-stationary prices
  is one of the most common beginner mistakes in financial ML.

  ADF null hypothesis: "this series has a unit root" (i.e. is NON-stationary).
  A small p-value -> reject the null -> the series IS stationary.

PART B: the multiple-testing trap, demonstrated rather than just described.
  If you test enough random, meaningless "signals" against returns, some
  fraction will show p < 0.05 purely by chance -- roughly 5% of them, by
  construction. This is why "I backtested 200 signal variations and found
  one with p=0.03" is close to worthless without a correction, and why
  NVDA's -0.082 raw-return autocorrelation from the previous step deserves
  skepticism rather than excitement: it's one result, but how many other
  lags/tickers/windows did we implicitly "test" before this one caught our
  eye?
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_report(series: pd.Series, label: str) -> str:
    series = series.dropna()
    stat, pvalue, *_ = adfuller(series)
    verdict = "STATIONARY" if pvalue < 0.05 else "NON-stationary (has a unit root)"
    return f"  {label:20s} ADF stat={stat:8.2f}  p-value={pvalue:.4f}  -> {verdict}"


def stationarity_section(panel: pd.DataFrame, ticker: str) -> str:
    grp = panel.loc[panel["ticker"] == ticker].sort_values("date")
    log_ret = np.log(grp["close"] / grp["close"].shift(1))
    lines = [f"-- Stationarity check: {ticker} --",
            adf_report(grp["close"], "raw price level"),
            adf_report(log_ret, "log returns")]
    return "\n".join(lines)


def multiple_testing_demo(n_fake_signals: int = 200, n_days: int = 2000, seed: int = 42) -> str:
    """
    Generate pure random noise as both a "return series" and N completely
    meaningless "signals" (also pure noise, with zero real relationship to
    returns). Count how many signals show a "significant" (p<0.05)
    correlation purely by chance.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n_days)

    from scipy import stats
    n_significant = 0
    best_p, best_i = 1.0, -1
    for i in range(n_fake_signals):
        fake_signal = rng.normal(0, 1, n_days)  # pure noise, NO real relationship to returns
        _, p = stats.pearsonr(fake_signal, returns)
        if p < 0.05:
            n_significant += 1
        if p < best_p:
            best_p, best_i = p, i

    expected = n_fake_signals * 0.05
    lines = [
        f"-- Multiple testing demo: {n_fake_signals} pure-noise 'signals' vs. pure-noise 'returns' --",
        f"  Signals with p < 0.05 by pure chance: {n_significant} "
        f"(expected ~{expected:.0f} at a 5% false-positive rate, since NONE of these signals are real)",
        f"  Best (most 'significant') fake signal: #{best_i}, p={best_p:.4f}",
        "",
        "  Lesson: if you test 200 feature/parameter combinations on one backtest and report",
        "  only the winner's p-value, you have NOT found a 5%-significance-level result --",
        "  you've guaranteed you'll find several by construction. This is why real research",
        "  logs, uses correction (e.g. Bonferroni, or better: strict train/validation/test splits",
        "  with the test set touched exactly once), and treats a single backtest's Sharpe ratio",
        "  with real suspicion until it survives out-of-sample data it never saw during design.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_ohlcv.csv"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"

    panel = pd.read_csv(path, parse_dates=["date"])
    print(stationarity_section(panel, ticker))
    print()
    print(multiple_testing_demo())
