"""Phase 5: LightGBM cross-sectional ranking model, purged + embargoed walk-forward.

Samples are (rebal_date t, ticker). Features: the 7 sector-neutral z-factors at t.
Label: percentile rank (0..1) of the 21-day forward return within date t, from
close[t+1] to close[t+22]. A sample's label window is therefore the trading-day
interval [t+1, t+22].

Outer loop (out-of-sample predictions): walk-forward by calendar year. Test
block = all rebalance dates in year Y (from FIRST_TEST_YEAR). Training = all
samples with t before the block, PURGED of any sample whose label window
overlaps the test block's span [first t + 1, last t + 22] - in practice the
last month before the block, whose label ends inside it.

Inner loop (hyperparameter choice, training data only): purged K-fold over
contiguous time blocks. Training samples whose label window overlaps the
validation span are purged, and samples starting within EMBARGO trading days
after the validation span are embargoed (serial correlation leaks backward
through the validation block into the samples right after it).

Baselines on the same out-of-sample dates:
  ew_prior   equal-weight composite of factors signed by PRIOR_SIGN (no fitting)
  ew_ic      equal-weight composite signed by the factor's training-period IC
  best_1     the single factor with the largest |training IC|, signed by it
  linear     OLS of the label on the 7 factors (training data)

Writes table predictions(rebal_date, ticker, <model columns>) and
results/model_*.csv.
"""
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import FACTORS, PRIOR_SIGN, RESULTS, connect, forward_returns, load_close, newey_west_t

H = 21
LAG = 1
FIRST_TEST_YEAR = 2019
INNER_FOLDS = 5
EMBARGO = 5
SEED = 7

GRID = [dict(num_leaves=nl, min_data_in_leaf=md, n_estimators=ne)
        for nl, md, ne in itertools.product([7, 15], [200, 1000], [100, 300])]
BASE_PARAMS = dict(objective="regression", learning_rate=0.05, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                   random_state=SEED, deterministic=True, n_jobs=4, verbose=-1)
MODELS = ["lgbm", "ew_prior", "ew_ic", "best_1", "linear"]


# ------------------------------------------------------------- purged splits
def purge_mask(start, end, test_lo, test_hi, embargo=0):
    """Boolean mask of samples allowed in training given a test span.

    start/end: label window of each sample (trading-day indices, inclusive).
    test_lo/test_hi: span covered by the test samples' label windows.
    A sample is dropped if its window overlaps [test_lo, test_hi] (purge) or it
    starts within `embargo` days after test_hi (embargo).
    """
    overlap = (start <= test_hi) & (end >= test_lo)
    embargoed = (start > test_hi) & (start <= test_hi + embargo)
    return ~(overlap | embargoed)


def label_windows(day_idx):
    """[entry, exit] trading-day indices of each sample's label: [t+1, t+1+H]."""
    return day_idx + LAG, day_idx + LAG + H


def purged_kfold(df, k=INNER_FOLDS, embargo=EMBARGO):
    """Yield (train_idx, val_idx) over contiguous date blocks, purged + embargoed."""
    dates = np.sort(df.day_idx.unique())
    blocks = np.array_split(dates, k)
    start, end = label_windows(df.day_idx.values)
    for b in blocks:
        val = df.day_idx.isin(b).values
        lo, hi = start[val].min(), end[val].max()
        train = purge_mask(start, end, lo, hi, embargo) & ~val
        yield np.where(train)[0], np.where(val)[0]


# ------------------------------------------------------------- helpers
def date_ic(df, col, label="fwd"):
    return df.groupby("rebal_date").apply(
        lambda g: g[col].rank().corr(g[label].rank()), include_groups=False)


def composite(df, signs):
    x = sum(df[f].fillna(0) * s for f, s in signs.items())
    return x / len(signs)


def fit_lgbm(train, params):
    m = lgb.LGBMRegressor(**BASE_PARAMS, **params)
    m.fit(train[FACTORS], train["label"])
    return m


def choose_params(train):
    scores = []
    for p in GRID:
        ics = []
        for tr, va in purged_kfold(train):
            m = fit_lgbm(train.iloc[tr], p)
            v = train.iloc[va].copy()
            v["pred"] = m.predict(v[FACTORS])
            ics.append(date_ic(v, "pred").mean())
        scores.append(np.mean(ics))
    best = int(np.argmax(scores))
    return GRID[best], scores


def load_panel(con):
    z = pd.read_sql("SELECT * FROM factors_z", con)
    days = pd.read_sql("SELECT date AS rebal_date, day_idx FROM trading_days", con)
    close = load_close(con)
    fwd = forward_returns(close, sorted(z.rebal_date.unique()), H, lag=LAG)
    df = z.merge(days, on="rebal_date").merge(fwd, on=["rebal_date", "ticker"], how="left")
    df["label"] = df.groupby("rebal_date")["fwd"].rank(pct=True)
    return df.sort_values(["day_idx", "ticker"]).reset_index(drop=True)


def walk_forward(df, shuffle_labels=False, purge=True, rng=None):
    """Out-of-sample predictions for every test year. Returns (preds, log)."""
    df = df.copy()
    df["year"] = df.rebal_date.str[:4].astype(int)
    labelled = df[df.label.notna()]
    preds, log = [], []
    for year in range(FIRST_TEST_YEAR, df.year.max() + 1):
        test = df[df.year == year].copy()
        past = labelled[labelled.day_idx < test.day_idx.min()]
        s, e = label_windows(past.day_idx.values)
        t_lo = test.day_idx.min() + LAG
        t_hi = test.day_idx.max() + LAG + H
        keep = purge_mask(s, e, t_lo, t_hi) if purge else np.ones(len(past), bool)
        train = past[keep].copy()
        _, tr_end = label_windows(train.day_idx.values)
        assert not purge or tr_end.max() < t_lo, "training label overlaps test period"
        if shuffle_labels:
            train["label"] = train.groupby("rebal_date")["label"].transform(
                lambda x: rng.permutation(x.values))

        params, scores = choose_params(train)
        m = fit_lgbm(train, params)
        test["lgbm"] = m.predict(test[FACTORS])

        tr_ic = {f: date_ic(train.dropna(subset=[f]), f, "label").mean() for f in FACTORS}
        test["ew_prior"] = composite(test, PRIOR_SIGN)
        test["ew_ic"] = composite(test, {f: np.sign(v) for f, v in tr_ic.items()})
        best = max(tr_ic, key=lambda f: abs(tr_ic[f]))
        test["best_1"] = test[best].fillna(0) * np.sign(tr_ic[best])

        X = np.column_stack([np.ones(len(train)), train[FACTORS].fillna(0).values])
        beta = np.linalg.lstsq(X, train["label"].values, rcond=None)[0]
        test["linear"] = np.column_stack([np.ones(len(test)), test[FACTORS].fillna(0).values]) @ beta

        log.append({"test_year": year, "n_train": len(train), "n_purged": int((~keep).sum()),
                    "train_last_date": train.rebal_date.max(), "test_first_date": test.rebal_date.min(),
                    "best_single": best, "best_single_sign": int(np.sign(tr_ic[best])),
                    **{f"p_{k}": v for k, v in params.items()},
                    "inner_cv_ic": max(scores),
                    **{f"gain_{f}": g for f, g in zip(FACTORS, m.booster_.feature_importance("gain"))}})
        preds.append(test[["rebal_date", "ticker", "fwd"] + MODELS])
    return pd.concat(preds, ignore_index=True), pd.DataFrame(log)


def score(preds):
    rows = {}
    lab = preds.dropna(subset=["fwd"])
    for mname in MODELS:
        ic = date_ic(lab, mname)
        dec = lab.assign(d=lab.groupby("rebal_date")[mname].transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)))
        spread = dec[dec.d == 9].groupby("rebal_date").fwd.mean() - dec[dec.d == 0].groupby("rebal_date").fwd.mean()
        rows[mname] = {"oos_mean_ic": ic.mean(), "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                       "hit_rate": (ic > 0).mean(), "n_months": len(ic),
                       "d10_d1_monthly": spread.mean(),
                       "d10_d1_t": spread.mean() / spread.std() * np.sqrt(len(spread))}
    return pd.DataFrame(rows).T


def main():
    con = connect()
    df = load_panel(con)
    preds, log = walk_forward(df)
    res = score(preds)

    # paired test: is LightGBM's monthly IC better than the equal-weight composite?
    lab = preds.dropna(subset=["fwd"])
    diff = date_ic(lab, "lgbm") - date_ic(lab, "ew_prior")
    res.loc["lgbm_minus_ew_prior", ["oos_mean_ic", "ic_t", "n_months"]] = [
        diff.mean(), newey_west_t(diff, 0), len(diff)]

    preds.drop(columns="fwd").to_sql("predictions", con, if_exists="replace", index=False)
    res.round(4).to_csv(RESULTS / "model_oos.csv", index_label="model")
    log.round(4).to_csv(RESULTS / "model_folds.csv", index=False)
    pd.set_option("display.width", 220)
    print(log[["test_year", "n_train", "n_purged", "train_last_date", "test_first_date",
               "best_single", "p_num_leaves", "p_min_data_in_leaf", "p_n_estimators",
               "inner_cv_ic"]].to_string(index=False))
    print("\nOut-of-sample (2019+):\n", res.round(4).to_string())
    gains = log[[f"gain_{f}" for f in FACTORS]].mean()
    print("\nMean LightGBM gain share:\n", (gains / gains.sum()).round(3).to_string())


if __name__ == "__main__":
    main()
