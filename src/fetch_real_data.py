"""
Milestone 1 — real data fetcher. RUN THIS ON YOUR OWN MACHINE, not in a
sandboxed environment without internet access to Yahoo Finance.

    pip install yfinance pandas pyarrow
    python src/fetch_real_data.py

Notes on the survivorship-bias trap in practice:
  yfinance (and most free sources) only know about tickers that exist TODAY.
  If you pull "S&P 500 constituents" from Wikipedia right now and backtest
  since 2015, you've already dropped every company that got acquired, went
  bankrupt, or was removed from the index -- the exact bias demonstrated in
  m1_survivorship_demo.py. For real research you'd want a point-in-time
  constituents list (e.g. from a paid vendor, or a maintained historical
  index-membership dataset) -- flag this as a known limitation of any
  free-data project, because interviewers will ask.
"""
from __future__ import annotations
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Run: pip install yfinance") from e


UNIVERSE = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "KO", "NVDA", "PG", "V", "HD"]


def fetch_universe(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV, split/dividend-adjusted, for a list of tickers."""
    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"WARNING: no data returned for {t} -- check ticker or date range")
            continue
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df.insert(0, "ticker", t)
        frames.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    panel = fetch_universe(UNIVERSE, start="2018-01-01", end="2025-12-31")
    panel.to_csv("data/real_ohlcv.csv", index=False)
    print(panel.groupby("ticker")["date"].agg(["min", "max", "count"]))
    print("\nRun src/data_quality.py against data/real_ohlcv.csv next -- "
          "same checks apply to real data as to simulated.")
