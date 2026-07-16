"""
Milestone 2, part 2 -- volatility clustering.

The core empirical fact: |return| and return^2 are autocorrelated even when
the raw return is not. This script proves that with two side-by-side ACF
checks and a formal Ljung-Box test, rather than just asserting it.

Ljung-Box test: tests the null hypothesis "these autocorrelations are all
zero" (i.e. no clustering). A small p-value -> reject the null -> real,
statistically significant autocorrelation is present.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


def rolling_annualized_vol(panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    panel["log_ret"] = panel.groupby("ticker")["close"].transform(lambda s: np.log(s / s.shift(1)))
    panel["roll_vol"] = panel.groupby("ticker")["log_ret"].transform(
        lambda s: s.rolling(window).std() * np.sqrt(252)
    )
    return panel


def clustering_report(panel: pd.DataFrame, ticker: str, lags: int = 10) -> str:
    r = panel.loc[panel["ticker"] == ticker, "log_ret"].dropna()

    acf_raw = acf(r, nlags=lags, fft=True)[1:]           # skip lag 0 (always 1.0)
    acf_sq = acf(r ** 2, nlags=lags, fft=True)[1:]

    lb_raw = acorr_ljungbox(r, lags=[lags], return_df=True)
    lb_sq = acorr_ljungbox(r ** 2, lags=[lags], return_df=True)

    lines = [f"{ticker} -- autocorrelation at lags 1-{lags}:",
            f"  raw returns:      {np.array2string(acf_raw, precision=3, suppress_small=True)}",
            f"  squared returns:  {np.array2string(acf_sq, precision=3, suppress_small=True)}",
            f"",
            f"  Ljung-Box (raw returns):     p-value = {lb_raw['lb_pvalue'].iloc[0]:.4f} "
            f"({'significant autocorrelation' if lb_raw['lb_pvalue'].iloc[0] < 0.05 else 'no significant autocorrelation -- consistent with weak-form efficiency'})",
            f"  Ljung-Box (squared returns):  p-value = {lb_sq['lb_pvalue'].iloc[0]:.2e} "
            f"({'significant autocorrelation -- THIS IS VOLATILITY CLUSTERING' if lb_sq['lb_pvalue'].iloc[0] < 0.05 else 'no significant clustering detected'})",
    ]
    return "\n".join(lines)


def plot_vol_clustering(panel: pd.DataFrame, ticker: str, outpath: str):
    grp = panel.loc[panel["ticker"] == ticker].sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(grp["date"], grp["log_ret"] * 100, linewidth=0.6, color="#2DD4A7")
    axes[0].set_ylabel("Daily log return (%)")
    axes[0].set_title(f"{ticker}: returns vs. 20d rolling volatility -- note the clustering")
    axes[0].axhline(0, color="gray", linewidth=0.5)

    axes[1].plot(grp["date"], grp["roll_vol"] * 100, linewidth=1.2, color="#8B7CFF")
    axes[1].set_ylabel("20d rolling ann. vol (%)")
    axes[1].fill_between(grp["date"], grp["roll_vol"] * 100, alpha=0.15, color="#8B7CFF")

    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()
    print(f"saved plot -> {outpath}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_ohlcv.csv"
    ticker = sys.argv[2] if len(sys.argv) > 2 else None

    panel = pd.read_csv(path, parse_dates=["date"])
    panel = rolling_annualized_vol(panel)

    tickers = [ticker] if ticker else panel["ticker"].unique().tolist()
    for tkr in tickers:
        print(clustering_report(panel, tkr))
        print()

    plot_vol_clustering(panel, tickers[0], f"notes/vol_clustering_{tickers[0]}.png")
