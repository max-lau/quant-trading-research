"""
Milestone 1 — data quality report.

Three checks every real research pipeline runs before touching a strategy:
  1. Coverage: did every ticker trade on every expected business day? Early
     terminations are exactly what survivorship bias hides.
  2. Gaps: missing business days inside a ticker's active range (holidays
     aside) that could indicate bad data, not just non-trading.
  3. Outliers: single-bar returns far outside normal range — often bad ticks,
     sometimes real (earnings, halts) but always worth a human look.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def quality_report(panel: pd.DataFrame, z_thresh: float = 6.0) -> str:
    lines = []
    full_range = pd.bdate_range(panel["date"].min(), panel["date"].max())

    lines.append(f"Universe: {panel['ticker'].nunique()} tickers, "
                 f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    lines.append(f"Expected business days in full range: {len(full_range)}\n")

    lines.append("-- Coverage & early terminations --")
    for tkr, grp in panel.groupby("ticker"):
        last_seen = grp["date"].max()
        pct_of_range = len(grp) / len(full_range) * 100
        flag = ""
        if last_seen < full_range[-5]:  # stopped trading well before the end of the sample
            flag = "  <-- FLAG: stopped trading early (possible delisting -- check for survivorship bias)"
        lines.append(f"  {tkr:12s} {len(grp):5d} rows  ({pct_of_range:5.1f}% of range)  "
                     f"last seen {last_seen.date()}{flag}")

    lines.append("\n-- Internal gaps (missing bdays before a ticker's last date) --")
    any_gaps = False
    for tkr, grp in panel.groupby("ticker"):
        active_range = pd.bdate_range(grp["date"].min(), grp["date"].max())
        missing = active_range.difference(grp["date"])
        if len(missing) > 0:
            any_gaps = True
            lines.append(f"  {tkr:12s} {len(missing)} missing bdays, e.g. {list(missing[:3].date)}")
    if not any_gaps:
        lines.append("  none detected")

    lines.append(f"\n-- Return outliers (|z| > {z_thresh}, by ticker) --")
    any_outliers = False
    for tkr, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date")
        rets = np.log(grp["close"] / grp["close"].shift(1))
        z = (rets - rets.mean()) / rets.std()
        flagged = grp.loc[z.abs() > z_thresh, "date"]
        if len(flagged) > 0:
            any_outliers = True
            lines.append(f"  {tkr:12s} {len(flagged)} bars flagged, e.g. {list(flagged.head(3).dt.date)}")
    if not any_outliers:
        lines.append("  none detected at this threshold")

    return "\n".join(lines)


if __name__ == "__main__":
    panel = pd.read_csv("data/real_ohlcv.csv", parse_dates=["date"])  # was read_parquet(simulated_ohlcv.parquet)
    print(quality_report(panel))
