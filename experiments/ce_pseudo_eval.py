"""Score the cross-encoder pseudo-labelling simulation on India's REAL labels (used only here).

For each cross-encoder variant (A = US only, pl = + India pseudo-labels, ctrl = + same amount of US data):
AUC on India's uncertain band, and India macro F0.5 after a 2nd-stage LightGBM [logit(p0), ce] on the band
(5-fold by S1, same as the ce2 A/B); best threshold per variant (relative comparison only).

python experiments/ce_pseudo_eval.py work/cepl_out/1/A/train_ce.parquet work/cepl_out/1/pl/train_ce.parquet ...
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from pseudo_sim import macro_f05

T = 0.7
ind = pl.read_parquet("work/cepl_india_labels.parquet").sort("i")
meta = pl.read_parquet("work/ab_data/tune_meta.parquet").join(ind.select("i").unique(), on="i")
band = ind["p0"].is_between(0.02, 0.98).to_numpy()
b = ind.filter(pl.col("p0").is_between(0.02, 0.98))
p = np.clip(b["p0"].to_numpy(), 1e-6, 1 - 1e-6)
lo, y, g = np.log(p / (1 - p)), b["y"].to_numpy(), b["i"].to_numpy()


def stacked(X):
    """India macro F0.5 with the band re-scored by a 2nd-stage model on X (out-of-fold by S1)."""
    p2 = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, g):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
        p2[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    prob = ind["p0"].to_numpy().copy()
    prob[band] = p2
    return macro_f05(ind, prob, meta)[0]


print(f"band {len(y)} pairs ({y.mean():.3f} true); p0 AUC {roc_auc_score(y, p):.4f}")
print(f"India F0.5: p0 alone @0.7 {macro_f05(ind, ind['p0'].to_numpy(), meta, thresholds=[T])[0]:.5f}, "
      f"control [p0] {stacked(lo[:, None]):.5f}")
for f in sys.argv[1:]:
    ce = b.join(pl.read_parquet(f), on=["i", "j"], how="left")["ce"].to_numpy()
    print(f"{f}: CE AUC {roc_auc_score(y, ce):.4f}  India F0.5 [p0, ce] {stacked(np.c_[lo, ce]):.5f}", flush=True)
