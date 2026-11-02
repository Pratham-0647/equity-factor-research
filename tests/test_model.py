import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model as M  # noqa: E402


def test_purge_mask_drops_overlap_and_embargo():
    start = np.array([0, 10, 20, 30, 40, 50])
    end = start + 21
    keep = M.purge_mask(start, end, test_lo=30, test_hi=45, embargo=5)
    # [10,31] and [20,41] overlap [30,45]; [30,51] and [40,61] are inside it;
    # 50 starts 5 days after 45 -> embargoed; 0..21 is clean
    assert keep.tolist() == [True, False, False, False, False, False]
    assert M.purge_mask(start, end, 30, 45, embargo=4).tolist()[-1]


def test_purged_kfold_no_leakage():
    days = np.repeat(np.arange(0, 21 * 40, 21), 5)       # 40 monthly dates x 5 stocks
    df = pd.DataFrame({"day_idx": days})
    s, e = M.label_windows(df.day_idx.values)
    seen = set()
    for tr, va in M.purged_kfold(df, k=5, embargo=5):
        assert not set(tr) & set(va)
        lo, hi = s[va].min(), e[va].max()
        assert not ((s[tr] <= hi) & (e[tr] >= lo)).any()
        assert not ((s[tr] > hi) & (s[tr] <= hi + 5)).any()
        seen |= set(va)
    assert seen == set(range(len(df)))                      # every sample validated once


def test_label_window_is_entry_to_exit():
    s, e = M.label_windows(np.array([100]))
    assert s[0] == 100 + M.LAG and e[0] == 100 + M.LAG + M.H
