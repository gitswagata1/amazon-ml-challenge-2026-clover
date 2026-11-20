"""Does LightGBM under-fit at full-data scale? Our PARAMS were tuned on the 400k-S1 table (~12M pairs); v9 trains on
2.1M S1 (44M pairs). CV (5-fold GroupKFold by S1) on an S1 sample of v9's saved features + extra columns, same
threshold grid / one-owner rule as run.py, for several tree counts / learning rates.

python experiments/capacity_ab.py --work work9 --n-s1 600000
Packed mode (another machine, no work9/dataset needed): --pack-out DIR writes DIR/features.parquet (i, j, y, features) and
DIR/truth.parquet (i, j) for the sampled S1; then --packed DIR [--folds 3 --arms base,2x] runs from those two files.
Learning curve (one fit per fold, scored after each tree count): --packed DIR --folds 5 --curve 1202,2404,3606
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import PARAMS, decide, f05, truth_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work9"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--n-s1", type=int, default=600000)
    ap.add_argument("--params", type=Path, default=Path("code/business_entity_resolution/best_params.json"))
    ap.add_argument("--pack-out", type=Path, help="write the sampled table + truth here and stop")
    ap.add_argument("--packed", type=Path, help="run from a packed folder (features.parquet, truth.parquet)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--arms", default="base,2x,2x_lr2,3x")
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--curve", help="tree counts, e.g. 1202,2404,3606: ONE model per fold with the largest count, scored after each count (float32, low memory)")
    a = ap.parse_args()
    t = json.loads(a.params.read_text())
    PARAMS.update(learning_rate=t["learning_rate"], num_leaves=t["num_leaves"], min_child_samples=t["min_child_samples"],
                  colsample_bytree=t["feature_fraction"], subsample=t["bagging_fraction"], reg_alpha=t["lambda_l1"],
                  reg_lambda=t["lambda_l2"], scale_pos_weight=t["scale_pos_weight"], n_estimators=t["n_estimators"])
    PARAMS["n_jobs"] = a.threads
    if a.packed:
        df, truth = pl.read_parquet(a.packed / "features.parquet"), pl.read_parquet(a.packed / "truth.parquet")
        ids = df["i"].unique()
    else:
        ids = pl.scan_parquet(a.work / "train_features.parquet").select("i").unique().collect()["i"].sample(a.n_s1, seed=0)
        df = pl.scan_parquet(a.work / "train_features.parquet").filter(pl.col("i").is_in(ids.implode())).collect()
        for pat in ("gpu/{s}_emb.parquet", "gpu/{s}_ce.parquet", "tfw_{s}.parquet", "rarity_{s}.parquet"):
            e = pl.scan_parquet(a.work / pat.format(s="train")).filter(pl.col("i").is_in(ids.implode())).collect()
            df = df.join(e, on=["i", "j"], how="left", maintain_order="left")
        truth = truth_pairs(a, str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")).filter(
            pl.col("i").is_in(ids.implode()))
        if a.pack_out:
            a.pack_out.mkdir(parents=True, exist_ok=True)
            df.write_parquet(a.pack_out / "features.parquet", compression="zstd")
            truth.write_parquet(a.pack_out / "truth.parquet")
            print(f"packed {len(ids)} S1, {df.height} pairs, {df.width - 3} features -> {a.pack_out}", flush=True)
            return
    if a.curve:  # learning curve: one fit per fold, predictions after k trees for each k
        p, y, g = df.select("i", "j"), df["y"].to_numpy(), df["i"].to_numpy()
        X = df.drop("i", "j", "y").cast(pl.Float32).to_numpy()  # float32 directly: no float64 copy
        del df
        ks = [int(k) for k in a.curve.split(",")]
        print(f"{len(ids)} S1, {p.height} pairs, {X.shape[1]} features, {a.folds} folds, curve {ks}", flush=True)
        oof = {k: np.zeros(p.height) for k in ks}
        for f, (tr, va) in enumerate(GroupKFold(a.folds).split(X, y, g)):
            m = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": max(ks)}).fit(X[tr], y[tr])
            for k in ks:
                oof[k][va] = m.predict_proba(X[va], num_iteration=k)[:, 1]
            print(f"  fold {f} done", flush=True)
        for k in ks:
            best = max((f05(decide(p, oof[k], th, True), truth, ids), round(float(th), 2)) for th in np.arange(0.5, 0.92, 0.02))
            print(f"trees {k:5d} lr {PARAMS['learning_rate']:.4f}: CV {best[0]:.5f} @ {best[1]}", flush=True)
        return
    p, y = df.select("i", "j"), df["y"].to_numpy()
    X = df.drop("i", "j", "y").to_pandas()
    del df
    print(f"{len(ids)} S1, {p.height} pairs, {X.shape[1]} features; base params {PARAMS}", flush=True)
    base_n, base_lr = PARAMS["n_estimators"], PARAMS["learning_rate"]
    arms = {"base": ("base", base_n, base_lr), "2x": ("2x trees", 2 * base_n, base_lr),
            "2x_lr2": ("2x trees, lr/2", 2 * base_n, base_lr / 2), "3x": ("3x trees", 3 * base_n, base_lr)}
    for name, n, lr in (arms[k] for k in a.arms.split(",")):
        oof = np.zeros(p.height)
        for tr, va in GroupKFold(a.folds).split(X, y, p["i"].to_numpy()):
            oof[va] = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": n, "learning_rate": lr}).fit(
                X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        best = max((f05(decide(p, oof, th, True), truth, ids), round(float(th), 2)) for th in np.arange(0.5, 0.92, 0.02))
        print(f"{name:16s} trees {n:5d} lr {lr:.4f}: CV {best[0]:.5f} @ {best[1]}", flush=True)


if __name__ == "__main__":
    main()
