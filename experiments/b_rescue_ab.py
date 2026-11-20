"""B test: bidirectional rescue. CV macro F0.5 (tuned LightGBM, A/B table, 5-fold by S1) using each S1's top-20
candidates, vs top-20 + pairs whose S1 is the candidate's best S1 (reverse rank 1 by rscore over ALL training S1,
taken from the full train_pairs_k30.parquet). Missed true pairs still count through tune_meta's ntrue.

python experiments/b_rescue_ab.py
"""
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from ab import TUNED
from tune import macro_f05
import lightgbm as lgb

D = "work/ab_data"
rr = (pl.read_parquet("work/train_pairs_k30.parquet", columns=["i", "j", "rscore"])
      .with_columns(rrank=pl.col("rscore").rank("ordinal", descending=True).over("j")).select("i", "j", "rrank"))
df = (pl.read_parquet(f"{D}/train_features.parquet")
      .drop([c for c in ("r_rel_c", "r_gap_c", "sup_sk", "sup_hn", "sup_both") if c in pl.read_parquet_schema(f"{D}/train_features.parquet")])
      .join(pl.read_parquet(f"{D}/train_tfidf.parquet"), on=["i", "j"], how="left")
      .join(pl.read_parquet(f"{D}/train_emb_v3pairs.parquet"), on=["i", "j"], how="left")
      .join(rr, on=["i", "j"], how="left").sort("i"))
meta = pl.read_parquet(f"{D}/tune_meta.parquet")
for name, keep in (("top-20", pl.col("brank") <= 20), ("top-20 + reverse-rank 1", (pl.col("brank") <= 20) | (pl.col("rrank") == 1))):
    d = df.filter(keep)
    X, y, g = d.drop("i", "j", "y", "rrank").to_numpy().astype(np.float32), d["y"].to_numpy(), d["i"].to_numpy()
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, g):
        oof[va] = lgb.LGBMClassifier(**TUNED).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    s, t = macro_f05(d, oof, meta)
    print(f"{name}: {d.height} pairs, {int(y.sum())} true, CV macro F0.5 = {s:.5f} @ t={t}", flush=True)
