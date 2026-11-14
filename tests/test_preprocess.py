import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocess as pp  # noqa: E402


def make_raw(seed=0, dates=("2020-01-31", "2020-02-28"), n=120):
    rng = np.random.default_rng(seed)
    rows = []
    sectors = np.array(["A"] * 50 + ["B"] * 40 + ["C"] * 28 + ["Solo"] + ["Duo"])
    for d in dates:
        x = rng.standard_t(3, n) * 10 + 5
        x[0] = 1e6                                      # extreme outlier
        rows.append(pd.DataFrame({"rebal_date": d, "ticker": [f"T{i}" for i in range(n)],
                                  "sector": sectors[:n], "f": x}))
    return pd.concat(rows, ignore_index=True)


def test_winsorize_bounds():
    x = pd.Series(np.arange(1000, dtype=float))
    w = pp.winsorize(x)
    assert w.min() == pytest.approx(x.quantile(0.01))
    assert w.max() == pytest.approx(x.quantile(0.99))


def test_sector_neutral_and_small_sectors_pooled():
    out = pp.process(make_raw(), factors=["f"])
    for _, g in out.groupby("rebal_date"):
        means = g.groupby("sector")["f"].mean()
        assert means[["A", "B", "C"]].abs().max() < 1e-12
        # Solo and Duo are pooled into "Other": not individually forced to zero
        assert g.loc[g.sector == "Solo", "f"].abs().item() > 0
        other = g[g.sector.isin(["Solo", "Duo"])]["f"]
        assert other.mean() == pytest.approx(0, abs=1e-12)


def test_each_date_is_independent():
    """Changing one date's data must not change another date's output."""
    raw = make_raw()
    base = pp.process(raw, factors=["f"])
    raw2 = raw.copy()
    raw2.loc[raw2.rebal_date == "2020-02-28", "f"] *= 7
    raw2.loc[raw2.rebal_date == "2020-02-28", "f"] += 100
    alt = pp.process(raw2, factors=["f"])
    jan = base.rebal_date == "2020-01-31"
    pd.testing.assert_series_equal(base.loc[jan, "f"], alt.loc[jan, "f"])


def test_affine_invariance_and_outlier_capped():
    raw = make_raw()
    a = pp.process(raw, factors=["f"])
    raw["f"] = raw["f"] * 3 - 50                     # z-score removes scale/shift
    b = pp.process(raw, factors=["f"])
    np.testing.assert_allclose(a["f"], b["f"], atol=1e-10)
    assert a["f"].abs().max() < 10                    # outlier no longer dominates


def test_missing_values_not_imputed():
    raw = make_raw()
    raw.loc[5, "f"] = np.nan
    out = pp.process(raw, factors=["f"])
    assert np.isnan(out.loc[5, "f"])
    assert out["f"].notna().sum() == raw["f"].notna().sum()
