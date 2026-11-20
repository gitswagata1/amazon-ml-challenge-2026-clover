"""Idea 2: importance-weight training rows toward an unseen country (covariate-shift adaptation), LOCO proxy.
Source country trains the matcher; the target country is 'unseen' (its FEATURES are used for the domain classifier,
its labels only for scoring). w = p/(1-p) of a domain classifier (source vs target pairs, out-of-fold for source
rows), clipped to [0.2, 5], optionally sqrt, mean 1. Same params / data for every arm; target F0.5 at the best
threshold and at 0.72.

python experiments/weight_loco.py
"""
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from run import PARAMS
from tune import macro_f05

D = "work/ab_data"
df = (pl.read_parquet(f"{D}/train_features.parquet")
      .drop([x for x in ("r_rel_c", "r_gap_c", "sup_sk", "sup_hn", "sup_both") if x in pl.read_parquet_schema(f"{D}/train_features.parquet")])
      .join(pl.read_parquet(f"{D}/train_tfidf.parquet"), on=["i", "j"], how="left")
      .join(pl.read_parquet(f"{D}/train_emb_v3pairs.parquet"), on=["i", "j"], how="left")
      .join(pl.scan_parquet("work/train_s1/*.parquet").select(pl.col("idx").alias("i"), "ctry").collect(), on="i").sort("i"))
meta = pl.read_parquet(f"{D}/tune_meta.parquet")
feats = [c for c in df.columns if c not in ("i", "j", "y", "ctry")]
DOM = dict(n_estimators=200, learning_rate=0.1, num_leaves=63, verbose=-1, n_jobs=-1)

for src, tgt in (("us", "india"), ("india", "us")):
    S, T = df.filter(pl.col("ctry") == src), df.filter(pl.col("ctry") == tgt)
    Xs, ys, Xt = S.select(feats).to_numpy().astype(np.float32), S["y"].to_numpy(), T.select(feats).to_numpy().astype(np.float32)
    mt = meta.join(T.select("i").unique(), on="i")
    # domain classifier: out-of-fold p(target | x) for source rows (2 folds), target rows subsampled to the source size
    rng = np.random.default_rng(0)
    tsub = Xt[rng.choice(len(Xt), min(len(Xt), len(Xs)), replace=False)]
    p = np.zeros(len(Xs))
    for (a, b), (c, d) in zip(KFold(2, shuffle=True, random_state=0).split(Xs), KFold(2, shuffle=True, random_state=0).split(tsub)):
        m = lgb.LGBMClassifier(**DOM).fit(np.vstack([Xs[a], tsub[c]]), np.r_[np.zeros(len(a)), np.ones(len(c))])
        p[b] = m.predict_proba(Xs[b])[:, 1]
    w = np.clip(p / (1 - p + 1e-9), 0.2, 5.0)
    print(f"{src}->{tgt}: domain p on source rows: mean {p.mean():.3f}, weights clipped at 5: {(w >= 5).mean():.3f}, at 0.2: {(w <= 0.2).mean():.3f}", flush=True)
    for name, sw in (("unweighted", None), ("w", w / w.mean()), ("sqrt w", np.sqrt(w) / np.sqrt(w).mean())):
        prob = lgb.LGBMClassifier(**PARAMS).fit(Xs, ys, sample_weight=sw).predict_proba(Xt)[:, 1]
        best = macro_f05(T, prob, mt)
        fixed = macro_f05(T, prob, mt, thresholds=[0.72])[0]
        print(f"  {name:10s}: {tgt} F0.5 best {best[0]:.5f} @ {best[1]}, @0.72 {fixed:.5f}", flush=True)
