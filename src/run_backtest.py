"""
Milestone 3, part 3 -- the honesty checks that separate real research from
a p-hacked backtest:

  1. COST SENSITIVITY: run each strategy at 0 bps (fantasy world) vs a
     realistic cost (5-10 bps for large-cap equities). If a strategy's
     Sharpe collapses when costs are added, its "edge" was mostly an
     artifact of trading too much for too little real signal -- exactly
     what happened historically to many naive mean-reversion strategies.

  2. WALK-FORWARD SPLIT: parameters (lookback windows here) must be chosen
     using ONLY the training period, then evaluated, untouched, on a test
     period that comes strictly AFTER the training period in time. This is
     the financial-ML equivalent of the "never touch the test set twice"
     rule -- and doing it wrong (e.g. shuffling dates into random folds)
     is one of the most common causes of a backtest that looks great in
     research and dies in production.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from backtest_engine import compute_strategy_returns, performance_stats, format_stats
from strategies import momentum_signal, mean_reversion_signal


def run_for_ticker(panel: pd.DataFrame, ticker: str, split_date: str = "2022-06-01"):
    grp = panel.loc[panel["ticker"] == ticker].sort_values("date").set_index("date")
    log_ret = np.log(grp["close"] / grp["close"].shift(1))

    signals = {
        "momentum_60d": momentum_signal(grp["close"], lookback=60),
        "mean_reversion_5d": mean_reversion_signal(grp["close"], lookback=5, z_entry=1.0),
    }

    print(f"\n===== {ticker} =====")
    for name, position in signals.items():
        print(f"\n-- {name} --")
        for cost_bps, cost_label in [(0.0, "0 bps (fantasy, no costs)"),
                                     (5.0, "5 bps (realistic large-cap cost)")]:
            bt = compute_strategy_returns(position, log_ret, cost_bps=cost_bps)
            stats = performance_stats(bt["net_ret"])
            print(format_stats(stats, cost_label))

        # walk-forward: were params (fixed here) "chosen" only seeing train,
        # then does the strategy still hold up on strictly later, unseen test data?
        train_ret = log_ret.loc[:split_date]
        test_ret = log_ret.loc[split_date:]
        train_pos = position.loc[:split_date]
        test_pos = position.loc[split_date:]

        bt_train = compute_strategy_returns(train_pos, train_ret, cost_bps=5.0)
        bt_test = compute_strategy_returns(test_pos, test_ret, cost_bps=5.0)
        stats_train = performance_stats(bt_train["net_ret"])
        stats_test = performance_stats(bt_test["net_ret"])
        print(f"  walk-forward (split {split_date}):")
        print(format_stats(stats_train, f"  train (<{split_date})"))
        print(format_stats(stats_test, f"  test  (>={split_date})"))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_ohlcv.csv"
    tickers = sys.argv[2:] if len(sys.argv) > 2 else None

    panel = pd.read_csv(path, parse_dates=["date"])
    tickers = tickers or panel["ticker"].unique().tolist()

    for tkr in tickers:
        run_for_ticker(panel, tkr)
