"""
Permanent regression test: proves compute_strategy_returns has no lookahead
bias and computes turnover/cost correctly. Run with: pytest tests/

If this test ever fails after an edit to backtest_engine.py, STOP and figure
out why before trusting any strategy result -- this is the single most
important correctness guarantee in the whole project.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from backtest_engine import compute_strategy_returns


def test_no_lookahead_and_cost_accounting():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price = pd.Series([100, 110, 121, 108.9, 120.0], index=dates)
    log_ret = np.log(price / price.shift(1))
    position = pd.Series([1, 1, -1, -1, 0], index=dates)

    bt = compute_strategy_returns(position, log_ret, cost_bps=0.0)

    # Each day's gross return must equal the PRIOR day's position times
    # THAT day's realized return -- never the same-day position.
    assert np.isclose(bt["gross_ret"].iloc[1], log_ret.iloc[1])
    assert np.isclose(bt["gross_ret"].iloc[2], log_ret.iloc[2])
    assert np.isclose(bt["gross_ret"].iloc[3], -log_ret.iloc[3])

    # Turnover: day4 flips position from +1 to -1 -> |change| = 2
    assert np.isclose(bt["turnover"].iloc[3], 2.0)
    # Day2 is the first real position change: from implicit 0 -> 1 -> |change|=1
    assert np.isclose(bt["turnover"].iloc[1], 1.0)


def test_costs_reduce_net_return_when_trading():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price = pd.Series([100, 110, 121, 108.9, 120.0], index=dates)
    log_ret = np.log(price / price.shift(1))
    position = pd.Series([1, 1, -1, -1, 0], index=dates)

    bt_free = compute_strategy_returns(position, log_ret, cost_bps=0.0)
    bt_costly = compute_strategy_returns(position, log_ret, cost_bps=50.0)  # exaggerated on purpose

    total_free = bt_free["net_ret"].sum()
    total_costly = bt_costly["net_ret"].sum()
    assert total_costly < total_free, "costs must reduce net return whenever turnover > 0"


if __name__ == "__main__":
    test_no_lookahead_and_cost_accounting()
    test_costs_reduce_net_return_when_trading()
    print("All backtest_engine tests passed.")
