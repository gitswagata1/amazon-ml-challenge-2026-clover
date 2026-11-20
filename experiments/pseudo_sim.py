"""Pseudo-labelling simulation for an unseen country (France is test-only): train on US, treat India as the unseen
country, add India's confident predictions as pseudo-labels, retrain, and measure India macro F0.5 with India's
REAL labels (used only for scoring). Threshold is chosen on US cross-validation, never on India labels.

python experiments/pseudo_sim.py --dir work/ab_data
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import PARAMS  # noqa: E402
from tune import macro_f05  # noqa: E402


def one_owner(df, prob, t):
    """Pairs accepted by the pipeline's decision rule (prob >= t and best S1 for the candidate)."""
    q = df.select("i", "j").with_columns(p=pl.Series(prob))
    return (q.filter((pl.col("p") == pl.col("p").max().over("j")) & (pl.col("p") >= t))
            .select("i", "j").with_columns(acc=pl.lit(True)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("work/ab_data"))
    a = ap.parse_args()
    df = (pl.read_parquet(a.dir / "train_features.parquet")
          .join(pl.read_parquet(a.dir / "train_tfidf.parquet"), on=["i", "j"], how="left")
          .join(pl.read_parquet(a.dir / "train_emb_v3pairs.parquet"), on=["i", "j"], how="left")
          .join(pl.scan_parquet("work/train_s1/*.parquet").select(pl.col("idx").alias("i"), "ctry").collect(), on="i")
          .sort("i"))
    meta = pl.read_parquet(a.dir / "tune_meta.parquet")
    us, ind = df.filter(pl.col("ctry") == "us"), df.filter(pl.col("ctry") == "india")
    feats = [c for c in df.columns if c not in ("i", "j", "y", "ctry")]
    Xu, yu, Xi = us.select(feats).to_numpy().astype(np.float32), us["y"].to_numpy(), ind.select(feats).to_numpy().astype(np.float32)
    mi = meta.join(ind.select("i").unique(), on="i")
    print(f"US {us.height} pairs / India {ind.height} pairs (India S1 {mi.height})", flush=True)

    # threshold from US cross-validation only
    oof = np.zeros(len(yu))
    for tr, va in GroupKFold(3).split(Xu, yu, us["i"].to_numpy()):
        oof[va] = lgb.LGBMClassifier(**PARAMS).fit(Xu[tr], yu[tr]).predict_proba(Xu[va])[:, 1]
    _, t = macro_f05(us, oof, meta.join(us.select("i").unique(), on="i"))
    print(f"threshold from US CV: {t}", flush=True)

    base = lgb.LGBMClassifier(**PARAMS).fit(Xu, yu)
    p0 = base.predict_proba(Xi)[:, 1]
    score = lambda p: macro_f05(ind, p, mi, thresholds=[t])[0]
    print(f"baseline (US-trained, India unseen): India F0.5 = {score(p0):.5f}", flush=True)
    print(f"oracle   (threshold tuned on India labels, reference only): {macro_f05(ind, p0, mi)[0]:.5f}", flush=True)

    for hi, lo, w, rounds in ((0.98, 0.02, 1.0, 1), (0.95, 0.05, 1.0, 1), (0.98, 0.02, 0.5, 1), (0.98, 0.02, 1.0, 2)):
        p = p0
        for _ in range(rounds):
            acc = ind.select("i", "j").join(one_owner(ind, p, t), on=["i", "j"], how="left")["acc"].fill_null(False).to_numpy()
            pos = acc & (p >= hi)            # confident matches that the decision rule would also accept
            neg = p <= lo                    # confident non-matches
            keep = pos | neg
            X = np.vstack([Xu, Xi[keep]])
            y = np.concatenate([yu, pos[keep].astype(int)])
            wt = np.concatenate([np.ones(len(yu)), np.full(keep.sum(), w)])
            p = lgb.LGBMClassifier(**PARAMS).fit(X, y, sample_weight=wt).predict_proba(Xi)[:, 1]
        true_pos = ind["y"].to_numpy()[pos].mean() if pos.any() else float("nan")
        print(f"pseudo hi={hi} lo={lo} w={w} rounds={rounds}: {pos.sum()} pos (precision {true_pos:.4f}), "
              f"{neg.sum()} neg -> India F0.5 = {score(p):.5f}", flush=True)


if __name__ == "__main__":
    main()
