"""
Permanent regression test: proves build_features never uses data from AFTER
the date it's computed for. Method: take two price series that are IDENTICAL
up through some date T, but differ afterward. Every feature value at or
before T must be identical between the two -- if it isn't, some feature is
peeking into the future.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import build_features


def test_features_have_no_lookahead():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(0)
    shared_history = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 70)))

    # Two "worlds" that agree on the first 70 days, then diverge completely.
    world_a = np.concatenate([shared_history, shared_history[-1] * np.exp(np.cumsum(rng.normal(0.05, 0.01, 30)))])
    world_b = np.concatenate([shared_history, shared_history[-1] * np.exp(np.cumsum(rng.normal(-0.05, 0.01, 30)))])

    close_a = pd.Series(world_a, index=dates)
    close_b = pd.Series(world_b, index=dates)

    feats_a = build_features(close_a)
    feats_b = build_features(close_b)

    # For every day up to and including day 70 (the shared history), every
    # feature must be IDENTICAL between world A and world B -- neither can
    # know which future the price is about to take.
    shared_period = feats_a.iloc[:70]
    shared_period_b = feats_b.iloc[:70]
    pd.testing.assert_frame_equal(shared_period, shared_period_b)


if __name__ == "__main__":
    test_features_have_no_lookahead()
    print("PASS: features are identical across diverging futures up to the shared history point.")
    print("      (this is the proof that no feature peeks into data that hadn't happened yet)")
