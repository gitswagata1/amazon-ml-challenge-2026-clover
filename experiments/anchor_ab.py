"""Task 2: anchor features. For each S1 the anchor a* = its candidate with the highest out-of-fold probability; every
pair (i, b) gets b's similarity to a* (name token_set, core ratio, address token_set), prob(a*), is_anchor and
same-source-as-a*. Targets "partly right" S1 (76% of v8's loss): true copies of a business resemble each other.
A/B table, tuned LightGBM, 5-fold by S1; OOF probs from the same folds (standard stacking). Control = 0.95287.

python experiments/anchor_ab.py
"""
import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from sklearn.model_selection import GroupKFold

from ab import TUNED
from features import _cp
from tune import macro_f05

D = "work/ab_data"
df = (pl.read_parquet(f"{D}/train_features.parquet")
      .drop([x for x in ("r_rel_c", "r_gap_c", "sup_sk", "sup_hn", "sup_both") if x in pl.read_parquet_schema(f"{D}/train_features.parquet")])
      .join(pl.read_parquet(f"{D}/train_tfidf.parquet"), on=["i", "j"], how="left")
      .join(pl.read_parquet(f"{D}/train_emb_v3pairs.parquet"), on=["i", "j"], how="left").sort("i", "j"))
meta = pl.read_parquet(f"{D}/tune_meta.parquet")
y, g = df["y"].to_numpy(), df["i"].to_numpy()
folds = list(GroupKFold(5).split(np.zeros(len(y)), y, g))


def cv(X):
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = lgb.LGBMClassifier(**TUNED).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return oof


X0 = df.drop("i", "j", "y").to_numpy().astype(np.float32)
oof = cv(X0)
print(f"control (recomputed): {macro_f05(df, oof, meta)}", flush=True)

c = pl.scan_parquet("work/train_cand/*.parquet").select(j="idx", name="name", core="core", addr="addr",
                                                          src=pl.col("entity_id").str.slice(0, 2)).collect()
p = df.select("i", "j").with_columns(p=pl.Series(oof))
anc = p.sort("p", descending=True).unique("i", keep="first").select("i", a="j", anc_prob="p")
p = p.join(anc, on="i").join(c, on="j").join(c.rename({"j": "a", "name": "an", "core": "ac", "addr": "aa", "src": "asrc"}), on="a")
p = p.sort("i", "j")
extra = np.column_stack([
    p["anc_prob"].to_numpy(),
    (p["j"] == p["a"]).to_numpy().astype(np.float32),
    _cp(fuzz.token_set_ratio, p["name"].to_list(), p["an"].to_list()),
    _cp(fuzz.ratio, p["core"].to_list(), p["ac"].to_list()),
    _cp(fuzz.token_set_ratio, p["addr"].to_list(), p["aa"].to_list()),
    (p["src"] == p["asrc"]).to_numpy().astype(np.float32),
]).astype(np.float32)
del p, c
oof2 = cv(np.hstack([X0, extra]))
print(f"+ anchor features: {macro_f05(df, oof2, meta)}  (bar: control + 0.001)", flush=True)
