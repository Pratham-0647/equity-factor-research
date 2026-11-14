import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import factors  # noqa: E402


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n, k = 400, 6
    days = pd.Index([f"d{i:04d}" for i in range(n)])
    m = rng.normal(0, 0.01, n)
    betas = np.linspace(0.5, 1.5, k)
    r = m[:, None] * betas + rng.normal(0, 0.015, (n, k))
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=days,
                         columns=[f"T{j}" for j in range(k)])
    close.iloc[50:60, 2] = np.nan           # a gap
    close.iloc[:150, 4] = np.nan            # a late listing
    mkt = pd.Series(1000 * np.cumprod(1 + m), index=days)
    mkt.iloc[200] = np.nan                  # index holiday
    return close, mkt


def test_no_lookahead(synthetic):
    """Appending future data must not change any factor value at or before t."""
    close, mkt = synthetic
    full = factors.compute_factor_panels(close, mkt)
    for t in (260, 300, 399):
        trunc = factors.compute_factor_panels(close.iloc[: t + 1], mkt.iloc[: t + 1])
        for name in full:
            pd.testing.assert_frame_equal(full[name].iloc[: t + 1], trunc[name],
                                          check_freq=False, obj=name)


def test_rolling_regression_matches_polyfit(synthetic):
    close, mkt = synthetic
    r = close.pct_change(fill_method=None)
    m = mkt.pct_change(fill_method=None)
    beta, resid = factors.rolling_regression(r, m, 60)
    t, col = 350, "T1"
    y, x = r[col].iloc[t - 59: t + 1], m.iloc[t - 59: t + 1]
    ok = y.notna() & x.notna()
    b, a = np.polyfit(x[ok], y[ok], 1)
    e = y[ok] - (a + b * x[ok])
    assert beta[col].iloc[t] == pytest.approx(b, rel=1e-8)
    assert resid[col].iloc[t] == pytest.approx(np.sqrt((e ** 2).sum() / (ok.sum() - 2)), rel=1e-8)


def test_definitions(synthetic):
    close, mkt = synthetic
    f = factors.compute_factor_panels(close, mkt)
    t, col = 300, "T0"
    c = close[col]
    assert f["mom_12_1"][col].iloc[t] == pytest.approx(c.iloc[t - 21] / c.iloc[t - 252] - 1)
    assert f["strev"][col].iloc[t] == pytest.approx(-(c.iloc[t] / c.iloc[t - 21] - 1))
    assert f["high_52w"][col].iloc[t] == pytest.approx(c.iloc[t] / c.iloc[t - 251: t + 1].max())
    vol = c.pct_change().iloc[t - 59: t + 1].std() * np.sqrt(252)
    assert f["lowvol"][col].iloc[t] == pytest.approx(1 / vol)


def test_insufficient_history_is_nan(synthetic):
    close, mkt = synthetic
    f = factors.compute_factor_panels(close, mkt)
    # T4 lists at row 150: no 252-day beta / momentum until well after that
    assert np.isnan(f["beta"]["T4"].iloc[300])
    assert np.isnan(f["mom_12_1"]["T4"].iloc[300])
    # gap rows carry no factor value
    assert f["lowvol"]["T2"].iloc[50:60].isna().all()
