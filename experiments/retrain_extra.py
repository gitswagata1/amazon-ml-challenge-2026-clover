"""v8.x: retrain v8's matcher with EXTRA feature columns, without recomputing v8's features.
Reads work/train_features.parquet (run.py train --save-features) and work/test_features/ (run.py predict
--save-test-features), left-joins each --extra file ({split} placeholder, columns i, j + features) on (i, j),
then the same 5-fold GroupKFold CV / threshold grid / one-owner rule / final fit as run.py train, and writes a
submission. Extra columns must be computed per split (fitted on train+test text where stats are involved).

python experiments/retrain_extra.py --params code/business_entity_resolution/best_params.json \
    --extra 'work/tfw_{split}.parquet' --out output_v81 --tag v81 [--seeds 3]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # in the submission zip: src/experiments/ -> src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))  # git repo
import run  # noqa: E402
from run import PARAMS, decide, f05, truth_pairs, write  # noqa: E402


def with_extra(df, patterns, split):
    for pat in patterns:
        e = pl.read_parquet(pat.format(split=split))
        df = df.join(e, on=["i", "j"], how="left", maintain_order="left")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--extra", action="append", default=[], help="parquet path with {split}; repeatable")
    ap.add_argument("--params", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="suffix for work/model_<tag>.txt, report_<tag>.json, oof_<tag>.parquet")
    ap.add_argument("--seeds", type=int, default=1, help="final model: average of N LightGBM seeds")
    a = ap.parse_args()
    if a.params:
        t = json.loads(a.params.read_text())
        PARAMS.update(learning_rate=t["learning_rate"], num_leaves=t["num_leaves"],
                      min_child_samples=t["min_child_samples"], colsample_bytree=t["feature_fraction"],
                      subsample=t["bagging_fraction"], reg_alpha=t["lambda_l1"], reg_lambda=t["lambda_l2"],
                      scale_pos_weight=t["scale_pos_weight"], n_estimators=t["n_estimators"])
    s1p, cp = str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")

    tr = with_extra(pl.read_parquet(a.work / "train_features.parquet"), a.extra, "train")
    p, y = tr.select("i", "j"), tr["y"].to_numpy()
    X = tr.drop("i", "j", "y")
    cols = X.columns
    X = X.to_pandas()
    del tr
    ids = p["i"].unique()
    truth = truth_pairs(a, s1p, cp).filter(pl.col("i").is_in(ids.implode()))
    print(f"[{a.tag}] {len(ids)} S1, {p.height} pairs, {len(cols)} features (+{', '.join(cols[-4:])})", flush=True)

    oof = np.zeros(p.height)
    for f, (itr, iva) in enumerate(GroupKFold(5).split(X, y, p["i"].to_numpy())):
        oof[iva] = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[itr], y[itr]).predict_proba(X.iloc[iva])[:, 1]
        print(f"  fold {f} done", flush=True)
    p.with_columns(y=pl.Series(y), oof=pl.Series(oof, dtype=pl.Float32)).write_parquet(a.work / f"oof_{a.tag}.parquet")
    best = max((f05(decide(p, oof, t, ex), truth, ids), round(float(t), 2), ex)
               for ex in (True, False) for t in np.arange(0.2, 0.96, 0.02))
    print(f"  CV macro F0.5 = {best[0]:.4f} @ t={best[1]} exclusive={best[2]}", flush=True)

    models = []
    for s in range(a.seeds):
        m = lgb.LGBMClassifier(**{**PARAMS, "random_state": s}).fit(X, y)
        m.booster_.save_model(str(a.work / f"model_{a.tag}_s{s}.txt"))
        models.append(m.booster_)
    del X
    report = {"cv_f05": best[0], "threshold": best[1], "exclusive": best[2], "features": cols, "seeds": a.seeds,
              "extra": a.extra, "n_train_s1": len(ids), "train_pairs": p.height}
    (a.work / f"report_{a.tag}.json").write_text(json.dumps(report, indent=2))

    ts1, tcp = str(a.work / "test_s1" / "*.parquet"), str(a.work / "test_cand" / "*.parquet")
    parts, probs = [], []
    for fp in sorted(glob.glob(str(a.work / "test_features" / "part-*.parquet"))):
        te = with_extra(pl.read_parquet(fp), a.extra, "test")
        Xt = te.select(cols).to_pandas()
        probs.append(np.mean([b.predict(Xt) for b in models], axis=0))
        parts.append(te.select("i", "j"))
    tp = pl.concat(parts)
    prob = np.concatenate(probs)
    tp.with_columns(prob=pl.Series(prob, dtype=pl.Float32)).write_parquet(a.work / f"test_prob_{a.tag}.parquet")
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "candidate_pairs.tsv", "candidate_entity_ids", ts1, tcp, tp)
    q = decide(tp, prob, best[1], best[2])
    write(a.out / "matching_results.tsv", "matched_entity_ids", ts1, tcp, q)
    print(f"[{a.tag}] wrote {a.out}: {q.height} matches, {q['i'].n_unique()} S1 with >=1 match", flush=True)


if __name__ == "__main__":
    run.PARAMS = PARAMS
    main()
