import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import risk as R  # noqa: E402


def test_max_drawdown():
    r = pd.Series([0.1, -0.5, 0.2])          # nav 1.1, 0.55, 0.66
    mdd, _ = R.max_drawdown(r)
    assert mdd == pytest.approx(-0.5)


def test_metrics_beta_and_sortino():
    rng = np.random.default_rng(0)
    idx = [f"2020-01-{i:02d}" for i in range(1, 2)] * 0
    dates = pd.date_range("2019-01-01", periods=1500, freq="B").strftime("%Y-%m-%d")
    mkt = pd.Series(rng.normal(0.0004, 0.01, 1500), index=dates)
    r = 0.5 * mkt + rng.normal(0.0002, 0.005, 1500)
    m = R.metrics(r, mkt)
    assert m["market_beta"] == pytest.approx(0.5, abs=0.03)
    ex = r
    sortino = ex.mean() / np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(252)
    assert m["sortino"] == pytest.approx(sortino)
    assert m["calmar"] == pytest.approx(m["cagr"] / abs(m["max_drawdown"]))


def test_expected_max_sr_matches_simulation():
    rng = np.random.default_rng(1)
    n, sd = 50, 0.4
    sim = (rng.normal(0, sd, (20000, n))).max(axis=1).mean()
    assert R.expected_max_sr(n, sd) == pytest.approx(sim, rel=0.03)
