"""Shared A/B harness: identical table, folds and metric for every matcher experiment, on any machine.

Table = v3 candidates, 400k training S1 (seed 0): features (Kaggle dataset kaggle-A/amizon-tuning)
+ TF-IDF cosines + e5 embedding cosines (kaggle-A/amizon-ab). Reference: LightGBM = 0.95008.

kaggle datasets download kaggle-A/amizon-tuning -p ab_data --unzip
kaggle datasets download kaggle-A/amizon-ab     -p ab_data --unzip
python experiments/ab.py --dir ab_data --model lgbm            # reference
python experiments/ab.py --dir ab_data --model catboost        # or: blend
python experiments/ab.py --dir ab_data --mix lgbm_tuned,catboost   # blends of saved OOFs
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import PARAMS  # noqa: E402
from tune import macro_f05  # noqa: E402


TUNED = dict(PARAMS, learning_rate=0.0406, num_leaves=286, min_child_samples=175, colsample_bytree=0.504,
             subsample=0.953, reg_alpha=0.096, reg_lambda=3.156, scale_pos_weight=1.21, n_estimators=1202)


def fit_predict(model, Xtr, ytr, Xva):
    if model in ("lgbm", "lgbm_tuned"):  # lgbm_tuned = Optuna best_params.json (Kaggle run, 32 trials)
        import lightgbm as lgb
        return lgb.LGBMClassifier(**(TUNED if model == "lgbm_tuned" else PARAMS)).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    if model == "catboost":  # Apache-2.0
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=1500, learning_rate=0.08, depth=8, verbose=0, thread_count=-1)
        return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    if model == "blend":
        return 0.5 * fit_predict("lgbm", Xtr, ytr, Xva) + 0.5 * fit_predict("catboost", Xtr, ytr, Xva)
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "lgbm_tuned", "catboost", "blend"])
    ap.add_argument("--mix", help="two saved OOF names, e.g. lgbm_tuned,catboost: score weighted blends")
    ap.add_argument("--extra", help="comma list of columns from <dir>/extra.parquet to add (tfidf_ab.py)")
    a = ap.parse_args()
    df = (pl.read_parquet(a.dir / "train_features.parquet")
          .drop([c for c in ("r_rel_c", "r_gap_c", "sup_sk", "sup_hn", "sup_both")
                 if c in pl.read_parquet_schema(a.dir / "train_features.parquet")])
          .join(pl.read_parquet(a.dir / "train_tfidf.parquet"), on=["i", "j"], how="left")
          .join(pl.read_parquet(a.dir / "train_emb_v3pairs.parquet"), on=["i", "j"], how="left").sort("i"))
    tag = a.model
    if a.extra:
        df = df.join(pl.read_parquet(a.dir / "extra.parquet", columns=["i", "j", *a.extra.split(",")]),
                     on=["i", "j"], how="left", maintain_order="left")
        tag += "+" + a.extra.replace(",", "+")
    meta = pl.read_parquet(a.dir / "tune_meta.parquet")
    if a.mix:  # score weighted averages of two saved OOF files, no refitting
        m1, m2 = a.mix.split(",")
        o1, o2 = np.load(a.dir / f"oof_{m1}.npy"), np.load(a.dir / f"oof_{m2}.npy")
        for w in (1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0):
            s, t = macro_f05(df, w * o1 + (1 - w) * o2, meta)
            print(f"{w:.1f}*{m1} + {1 - w:.1f}*{m2}: CV macro F0.5 = {s:.5f} @ t={t}", flush=True)
        return
    X, y, g = df.drop("i", "j", "y").to_numpy().astype(np.float32), df["y"].to_numpy(), df["i"].to_numpy()
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, g)):
        oof[va] = fit_predict(a.model, X[tr], y[tr], X[va])
        print(f"fold {k} done", flush=True)
    np.save(a.dir / f"oof_{tag}.npy", oof)
    s, t = macro_f05(df, oof, meta)
    print(f"{tag}: features={X.shape[1]}  CV macro F0.5 = {s:.5f} @ t={t}  (reference lgbm 0.95008)")


if __name__ == "__main__":
    main()
