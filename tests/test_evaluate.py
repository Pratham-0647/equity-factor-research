import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import forward_returns, newey_west_t  # noqa: E402
from evaluate import decile_returns, monotonicity, rank_ic  # noqa: E402


def test_forward_returns_alignment_and_no_truncation():
    idx = pd.Index([f"d{i:02d}" for i in range(30)])
    close = pd.DataFrame({"A": np.arange(1, 31, dtype=float)}, index=idx)
    f = forward_returns(close, ["d00", "d05", "d20"], horizon=5, lag=1)
    # d00: enter at d01 (2), exit d06 (7)
    assert f.set_index("rebal_date").loc["d00", "fwd"] == pytest.approx(7 / 2 - 1)
    # d20 + 1 + 5 = 26 < 30 -> present; a date whose exit is past the end is dropped
    with pytest.raises(ValueError):
        forward_returns(close, ["nope"], 5)
    assert "d25" not in set(forward_returns(close, ["d20", "d25"], 5).rebal_date)


def test_rank_ic_matches_scipy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"rebal_date": np.repeat(["a", "b"], 50),
                       "f": rng.normal(size=100), "fwd": rng.normal(size=100)})
    ic = rank_ic(df, ["f"])
    for d in ["a", "b"]:
        g = df[df.rebal_date == d]
        assert ic.loc[d, "f"] == pytest.approx(spearmanr(g.f, g.fwd)[0])


def test_newey_west_zero_lags_is_naive_t():
    x = np.random.default_rng(1).normal(0.1, 1, 200)
    naive = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_t(x, 0) == pytest.approx(naive)


def test_deciles_and_monotonicity():
    n = 200
    df = pd.DataFrame({"rebal_date": "a", "f": np.arange(n, dtype=float)})
    df["fwd"] = df.f / 1000
    d = decile_returns(df, ["f"])
    v = d[list(range(1, 11))].iloc[0].values
    assert np.all(np.diff(v) > 0)
    rho, steps = monotonicity(v)
    assert rho == pytest.approx(1) and steps == 1
