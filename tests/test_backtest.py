import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import backtest as B  # noqa: E402


@pytest.fixture
def toy():
    idx = pd.Index([f"d{i}" for i in range(6)])
    # A +10%/day, B -10%/day after day 0
    close = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41, 161.051],
                          "B": [100, 90, 81, 72.9, 65.61, 59.049]}, index=idx)
    return B.daily_returns(close)


def test_long_only_no_cost_compounds_exactly(toy):
    # signal d0 -> trade at close d1 -> earns d2..d5 on A
    bt = B.run({"d0": pd.Series({"A": 1.0})}, toy, cost_bps=0)
    assert bt.index[0] == "d1"
    assert (1 + bt.ret).prod() == pytest.approx(1.1 ** 4)


def test_long_short_and_cost_on_turnover(toy):
    w = pd.Series({"A": 1.0, "B": -1.0})
    bt = B.run({"d0": w}, toy, cost_bps=10)
    assert bt.turnover.iloc[0] == pytest.approx(2.0)          # from cash to +1/-1
    assert bt.cost.iloc[0] == pytest.approx(2.0 * 10 / 1e4)
    # day 2: long +10%, short gains 10% on -1 => +20% of (1 - cost)
    assert bt.ret.iloc[1] == pytest.approx(0.2)


def test_drift_turnover_when_target_unchanged(toy):
    w = pd.Series({"A": 0.5, "B": 0.5})
    bt = B.run({"d0": w, "d2": w}, toy, cost_bps=0)
    # at d3 (trade for d2 signal) weights drifted away from 50/50 -> nonzero turnover
    t = bt.loc["d3", "turnover"]
    assert 0 < t < 0.5


def test_target_weights_shapes():
    sig = pd.DataFrame({"ticker": list("ABCDEFGHIJ"), "s": np.arange(10.0)})
    ls = B.target_weights(sig, "s", "ls", q=0.2)
    assert ls.sum() == pytest.approx(0) and ls.abs().sum() == pytest.approx(2)
    assert set(ls[ls > 0].index) == {"I", "J"} and set(ls[ls < 0].index) == {"A", "B"}
    lo = B.target_weights(sig, "s", "lo", scheme="vol", q=0.2,
                          vol=pd.Series({"I": 0.2, "J": 0.4}))
    assert lo["I"] == pytest.approx(2 / 3) and lo.sum() == pytest.approx(1)


def test_breakeven_interpolation():
    assert B.breakeven(np.array([0, 10, 20]), np.array([0.1, 0.05, -0.05])) == pytest.approx(15)
