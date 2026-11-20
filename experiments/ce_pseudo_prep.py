"""Data for the cross-encoder pseudo-labelling simulation (India plays the unseen country, like France on test).

Writes <out>/:
  us_train.parquet  i, j, y      cross-encoder training pairs of US S1 only (from ce_train.parquet)
  india.parquet     i, j, y, p0, acc   India pairs of the A/B table; p0 = LightGBM trained on US only,
                                  acc = accepted by threshold (from US CV) + one-owner rule
India's labels are used only for scoring in ce_pseudo_eval.py, never for training.

python experiments/ce_pseudo_prep.py --dir work/ab_data --ce-train work/kaggle_ce_data/ce_train.parquet --out work/kaggle_cepl_data
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from pseudo_sim import PARAMS, macro_f05, one_owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("work/ab_data"))
    ap.add_argument("--ce-train", type=Path, default=Path("work/kaggle_ce_data/ce_train.parquet"))
    ap.add_argument("--out", type=Path, default=Path("work/kaggle_cepl_data"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ctry = pl.scan_parquet("work/train_s1/*.parquet").select(pl.col("idx").alias("i"), "ctry").collect()

    us_tr = pl.read_parquet(a.ce_train).join(ctry, on="i").filter(pl.col("ctry") == "us").drop("ctry")
    us_tr.write_parquet(a.out / "us_train.parquet")
    print(f"us_train: {us_tr.height} pairs, {us_tr['y'].sum()} true, {us_tr['i'].n_unique()} S1", flush=True)

    df = (pl.read_parquet(a.dir / "train_features.parquet")
          .join(pl.read_parquet(a.dir / "train_tfidf.parquet"), on=["i", "j"], how="left")
          .join(pl.read_parquet(a.dir / "train_emb_v3pairs.parquet"), on=["i", "j"], how="left")
          .join(ctry, on="i").sort("i"))
    meta = pl.read_parquet(a.dir / "tune_meta.parquet")
    us, ind = df.filter(pl.col("ctry") == "us"), df.filter(pl.col("ctry") == "india")
    feats = [c for c in df.columns if c not in ("i", "j", "y", "ctry")]
    Xu, yu = us.select(feats).to_numpy().astype(np.float32), us["y"].to_numpy()
    oof = np.zeros(len(yu))
    for tr, va in GroupKFold(3).split(Xu, yu, us["i"].to_numpy()):
        oof[va] = lgb.LGBMClassifier(**PARAMS).fit(Xu[tr], yu[tr]).predict_proba(Xu[va])[:, 1]
    _, t = macro_f05(us, oof, meta.join(us.select("i").unique(), on="i"))
    p0 = lgb.LGBMClassifier(**PARAMS).fit(Xu, yu).predict_proba(ind.select(feats).to_numpy().astype(np.float32))[:, 1]
    out = ind.select("i", "j", "y").with_columns(p0=pl.Series(p0, dtype=pl.Float32))
    out = (out.join(one_owner(out, p0, t), on=["i", "j"], how="left").with_columns(pl.col("acc").fill_null(False)))
    out.write_parquet(a.out / "india.parquet")
    print(f"india: {out.height} pairs, threshold {t}, band [0.02, 0.98]: "
          f"{out.filter(pl.col('p0').is_between(0.02, 0.98)).height}", flush=True)


if __name__ == "__main__":
    main()
