"""LightGBM matcher with extra feature columns.

Reads the saved feature tables (run.py train / predict --features-only), left-joins each --extra file ({split}
placeholder, columns i, j + features) on (i, j), runs 5-fold GroupKFold cross-validation by Source 1 record, picks the
threshold and one-to-one rule that maximise macro F0.5 on out-of-fold predictions, fits the final model (average of
--seeds seeds) and scores the test pairs.

python pipeline/classifier.py --work work10 --params best_params.json --extra 'work10/gpu/{split}_emb.parquet' \
    --extra 'work10/rarity_{split}.parquet' --seeds 3 --out output_v10 --tag v10
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

_R = Path(__file__).resolve().parents[1]  # src/
sys.path.insert(0, str(_R / "code/business_entity_resolution/src" if (_R / "code").exists() else _R))
import run  # noqa: E402
from run import PARAMS, decide, f05, truth_pairs, write  # noqa: E402


def with_extra(df, patterns, split, allow_missing=False):
    """Left-join each extra file on (i, j), reading only the rows of df's i-range. Stops on duplicate (i, j) rows in
    an extra file, and on pairs of df that it doesn't cover (unless allow_missing: they become NaN)."""
    lo, hi = df["i"].min(), df["i"].max()
    for pat in patterns:
        f = pat.format(split=split)
        e = (pl.scan_parquet(f).filter(pl.col("i").is_between(lo, hi))
             .join(df.select("i", "j").lazy(), on=["i", "j"], how="semi").collect())
        dup = e.height - e.select("i", "j").n_unique()
        assert dup == 0, f"{f}: {dup} duplicate (i, j) rows"
        missing = df.height - e.height
        if missing:
            msg = f"{f}: {missing} of {df.height} pairs have no row (their columns would be NaN)"
            assert allow_missing, msg + "; recompute the file on these pairs or pass --allow-missing"
            print("  warning:", msg, flush=True)
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
    ap.add_argument("--allow-missing", action="store_true", help="extra files may miss pairs (NaN) instead of stopping")
    a = ap.parse_args()
    if a.params:
        t = json.loads(a.params.read_text())
        PARAMS.update(learning_rate=t["learning_rate"], num_leaves=t["num_leaves"],
                      min_child_samples=t["min_child_samples"], colsample_bytree=t["feature_fraction"],
                      subsample=t["bagging_fraction"], reg_alpha=t["lambda_l1"], reg_lambda=t["lambda_l2"],
                      scale_pos_weight=t["scale_pos_weight"], n_estimators=t["n_estimators"])
    s1p, cp = str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")

    tr = with_extra(pl.read_parquet(a.work / "train_features.parquet"), a.extra, "train", a.allow_missing)
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
    print(f"  CV macro F0.5 = {best[0]:.4f} @ t={best[1]} exclusive={best[2]} (S1 with >= 1 candidate)", flush=True)
    meta = a.work / "tune_meta.parquet"  # every S1 of run.py's training sample, also those without any candidate
    if meta.exists():  # same S1 set as run.py train's CV, so the numbers compare across versions; the threshold
        m = pl.read_parquet(meta)  # is unchanged (S1 without candidates score the same at every threshold)
        n0, t0 = m.height - len(ids), int((m.filter(~pl.col("i").is_in(ids.implode()))["ntrue"] == 0).sum())
        cv_all = (best[0] * len(ids) + t0) / m.height
        print(f"  CV macro F0.5 = {cv_all:.4f} over all {m.height} sampled S1 (run.py's definition; "
              f"{n0} without candidates, {t0} of them singletons)", flush=True)

    models = []
    for s in range(a.seeds):
        m = lgb.LGBMClassifier(**{**PARAMS, "random_state": s}).fit(X, y)
        m.booster_.save_model(str(a.work / f"model_{a.tag}_s{s}.txt"))
        models.append(m.booster_)
    del X
    report = {"cv_f05": best[0], "cv_f05_all_s1": cv_all if meta.exists() else None, "threshold": best[1],
              "exclusive": best[2], "features": cols, "seeds": a.seeds,
              "extra": a.extra, "n_train_s1": len(ids), "train_pairs": p.height}
    (a.work / f"report_{a.tag}.json").write_text(json.dumps(report, indent=2))

    ts1, tcp = str(a.work / "test_s1" / "*.parquet"), str(a.work / "test_cand" / "*.parquet")
    parts, probs = [], []
    for fp in sorted(glob.glob(str(a.work / "test_features" / "part-*.parquet"))):
        te = with_extra(pl.read_parquet(fp), a.extra, "test", a.allow_missing)
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
