"""
Milestone 2, part 1 — return distributions.

Quick refresher (since this is the part interviews probe hardest):

Simple return:  r_t = P_t / P_{t-1} - 1
Log return:     l_t = ln(P_t / P_{t-1})

Why log returns for research:
  - They're additive across time: l_1 + l_2 = total log return over 2 periods.
    Simple returns are NOT additive (you have to compound them: (1+r1)(1+r2)-1).
  - For small moves, log and simple returns are nearly identical (l ~ r when r is
    small), so this convenience costs you almost nothing in practice.
  - Annualization: multiply mean log return by 252 (trading days/year), and
    volatility (std) by sqrt(252) -- because variance of independent sums
    scales linearly with time, so std scales with sqrt(time).

Fat tails: real return distributions have far more extreme moves than a normal
distribution predicts. We measure this with:
  - Skewness: asymmetry. Equity returns are typically negatively skewed
    (crashes are sharper than rallies).
  - Kurtosis: "tailedness". Normal distribution has kurtosis = 3 (or "excess
    kurtosis" = 0 in the common convention). Real markets show excess kurtosis
    often in the 3-10+ range -- meaning extreme days are FAR more common than
    a Gaussian model would ever predict.

This is precisely why risk models that assume normality (like naive VaR) blow
up during crashes: the crash IS the fat tail the model didn't budget for.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def compute_log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    panel["log_ret"] = panel.groupby("ticker")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return panel


def distribution_report(panel: pd.DataFrame) -> str:
    lines = []
    lines.append(f"{'ticker':8s} {'ann.ret':>9s} {'ann.vol':>9s} {'skew':>8s} "
                 f"{'excess kurt':>12s} {'worst day':>10s} {'best day':>10s}")
    for tkr, grp in panel.groupby("ticker"):
        r = grp["log_ret"].dropna()
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        skew = stats.skew(r)
        kurt = stats.kurtosis(r)  # scipy's default IS excess kurtosis (normal -> 0)
        lines.append(f"{tkr:8s} {ann_ret:+8.1%} {ann_vol:9.1%} {skew:+8.2f} "
                     f"{kurt:12.2f} {r.min():+9.1%} {r.max():+9.1%}")

    lines.append("\nInterpretation:")
    lines.append("  - excess kurtosis > 0 means fatter tails than a normal distribution")
    lines.append("    (extreme days are more frequent/severe than Gaussian VaR would predict)")
    lines.append("  - negative skew means the LEFT tail (crashes) is fatter than the right")
    lines.append("    -- consistent with 'markets fall faster than they rise'")
    return "\n".join(lines)


def normality_test(panel: pd.DataFrame, ticker: str) -> str:
    """Jarque-Bera test: is this return series consistent with a normal distribution?"""
    r = panel.loc[panel["ticker"] == ticker, "log_ret"].dropna()
    jb_stat, jb_p = stats.jarque_bera(r)
    verdict = "REJECT normality" if jb_p < 0.05 else "fail to reject normality"
    return (f"{ticker}: Jarque-Bera stat={jb_stat:.1f}, p-value={jb_p:.2e} -> {verdict}\n"
           f"  (p < 0.05 is the near-universal case for daily equity returns --\n"
           f"   this is not a bug in the test, it's the market telling you it isn't Gaussian)")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_ohlcv.csv"
    panel = pd.read_csv(path, parse_dates=["date"])
    panel = compute_log_returns(panel)
    print(distribution_report(panel))
    print()
    for tkr in panel["ticker"].unique():
        print(normality_test(panel, tkr))
