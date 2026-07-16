"""
Milestone 3, part 1 -- the backtesting engine everything else builds on.

THE SINGLE MOST IMPORTANT LINE IN THIS FILE:

    strategy_ret = position.shift(1) * asset_ret

Why shift(1): "position" represents the size/direction you decided to hold
AFTER seeing today's close (i.e. your signal is computed using data through
today). You cannot earn today's return with a decision made using today's
own closing price -- that would be look-ahead bias. You can only earn
TOMORROW's return with a position sized using information available as of
today's close. This one-line detail is the most common bug in amateur
backtests, and it single-handedly turns a mediocre strategy into a
"genius" one if you get it backwards. If an interviewer asks you to walk
through a backtest, this is the line they're listening for.

Transaction costs: charged on TURNOVER (the change in position), not on
the position itself -- you don't pay a cost for holding, only for trading.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_strategy_returns(position: pd.Series, asset_log_ret: pd.Series,
                              cost_bps: float = 5.0) -> pd.DataFrame:
    """
    position: target position at each date, decided using info through that
        date's close (e.g. -1/0/+1, or continuous). Index must align with
        asset_log_ret's index.
    asset_log_ret: the asset's log return realized ON each date.
    cost_bps: round-trip-style cost charged per unit of position CHANGE,
        in basis points (5 bps = 0.05%, a reasonable large-cap equity estimate
        covering spread + slippage; tighten/loosen per instrument in practice).

    Returns a DataFrame with the no-lookahead strategy return, cost drag,
    and net return per bar.
    """
    position = position.reindex(asset_log_ret.index).fillna(0.0)
    lagged_position = position.shift(1).fillna(0.0)          # <-- the critical line

    turnover = lagged_position.diff().abs().fillna(lagged_position.abs())
    cost = turnover * (cost_bps / 1e4)

    gross_ret = lagged_position * asset_log_ret
    net_ret = gross_ret - cost

    return pd.DataFrame({
        "position": lagged_position,
        "turnover": turnover,
        "gross_ret": gross_ret,
        "cost": cost,
        "net_ret": net_ret,
    })


def performance_stats(net_ret: pd.Series, periods_per_year: int = 252) -> dict:
    r = net_ret.dropna()
    if len(r) == 0 or r.std() == 0:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
               "max_drawdown": 0.0, "calmar": 0.0, "win_rate": 0.0}

    equity = (1 + r).cumprod()
    ann_return = equity.iloc[-1] ** (periods_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() / r.std()) * np.sqrt(periods_per_year)

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    nonzero = r[r != 0]
    win_rate = (nonzero > 0).mean() if len(nonzero) else 0.0

    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
           "max_drawdown": max_dd, "calmar": calmar, "win_rate": win_rate}


def format_stats(stats: dict, label: str) -> str:
    return (f"  {label:28s} ann.ret={stats['ann_return']:+7.1%}  "
           f"ann.vol={stats['ann_vol']:6.1%}  sharpe={stats['sharpe']:+5.2f}  "
           f"max_dd={stats['max_drawdown']:+7.1%}  calmar={stats['calmar']:+5.2f}  "
           f"win_rate={stats['win_rate']:5.1%}")
