"""End-to-end pipeline: normalise -> block -> features -> LightGBM -> threshold -> output/*.tsv

python code/business_entity_resolution/src/run.py all --data dataset --work work --out output
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from blocking import block, cand_context, ensure
from features import features

PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
CHUNK = 1_000_000


def atomic(path, write):
    """write(tmp_path), then rename onto path: a crash or a parallel reader never sees a half-written cache file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, path)


def pairs(a, split):
    """-> (s1 glob, cand glob, LazyFrame of blocking pairs joined with candidate context)."""
    s1p, cp = ensure(a.data, a.work, split, "s1"), ensure(a.data, a.work, split, "cand")
    # non-default blocking options go into the cache name, so a rebuild never silently reuses other pairs
    tag = ((f"_p{a.pool}" if a.pool != 600 else "") + ("_ap" if a.addr_partial else "") + ("_ak" if a.addr_keys else "")
           + (f"_bs{a.n_train}" if a.block_sample and split == "train" else ""))
    path = a.work / f"{split}_pairs_k{a.k}{tag}.parquet"
    if not path.exists():  # --block-sample: only the training sample's S1 (fast CV A/B of blocking settings)
        only = train_sample(a, s1p) if a.block_sample and split == "train" else None
        block(s1p, cp, path, k=a.k, pool=a.pool, only=only, addr_keys=a.addr_keys, addr_partial=a.addr_partial)
    knn = a.knn / f"{split}_knn.parquet" if a.knn else None
    knn2 = a.knn2 / f"{split}_knn.parquet" if getattr(a, "knn2", None) else None  # fine-tuned retriever (ret_train.py)
    if a.k_use and a.k_use < a.k:  # keep each S1's top-k_use candidates (smaller candidate set for the matcher)
        small = path.with_name(f"{path.stem}_use{a.k_use}{'_rs' if a.rescue else ''}"
                               f"{f'_knn{a.knn_top}' if knn else ''}{f'_rk{a.knn2_top}' if knn2 else ''}"
                               f"{'_re' if a.rare_empty else ''}.parquet")
        args = {"knn": str(knn.resolve()) if knn else None, "knn_top": a.knn_top if knn else None,
                "rescue": a.rescue, "rare_empty": a.rare_empty}
        if knn2:
            args |= {"knn2": str(knn2.resolve()), "knn2_top": a.knn2_top}
        side = small.with_suffix(".args.json")  # the name can't hold the kNN file: never reuse another one's pairs
        if small.exists() and side.exists():
            assert json.loads(side.read_text()) == args, (
                f"{small.name} was built with {side.read_text()}, now {args}: use another --work or delete it")
        elif small.exists():  # cache from before this check: record what it is assumed to be
            print(f"  note: {small.name} has no {side.name}; assuming it was built with {args}", flush=True)
            side.write_text(json.dumps(args))
        if not small.exists():
            lp, keep = pl.scan_parquet(path), pl.col("brank") <= a.k_use
            if a.rescue:  # bidirectional rescue: also keep each candidate's best S1 by re-rank score (b_rescue_ab.py)
                lp = lp.with_columns(rrank=pl.col("rscore").rank("ordinal", descending=True).over("j"))
                keep = keep | (pl.col("rrank") == 1)
            kept = lp.filter(keep).drop("rrank", strict=False).collect()
            if knn:  # + e5 same-country nearest neighbours that blocking missed; their pool scores stay null (NaN)
                nn = (pl.read_parquet(knn, columns=["i", "j", "knn_rank"]).filter(pl.col("knn_rank") < a.knn_top)
                      .select("i", "j").join(kept.select("i", "j"), on=["i", "j"], how="anti"))
                kept = pl.concat([kept, nn], how="diagonal_relaxed")
            if knn2:  # + fine-tuned retriever neighbours (same country) not found so far
                nn = (pl.read_parquet(knn2, columns=["i", "j", "knn_rank"]).filter(pl.col("knn_rank") < a.knn2_top)
                      .select("i", "j").join(kept.select("i", "j"), on=["i", "j"], how="anti"))
                kept = pl.concat([kept, nn], how="diagonal_relaxed")
            if a.rare_empty:  # a name unique among the country's S1 whose exact twin is a rare candidate with NO address
                rp = rare_empty_pairs(s1p, cp)  # (blocking can't use the address; 97.6% true on train)
                kept = (pl.concat([kept, rp.join(kept.select("i", "j"), on=["i", "j"], how="anti")], how="diagonal_relaxed")
                        .join(rp.with_columns(rare_empty=pl.lit(1, pl.UInt8)), on=["i", "j"], how="left")
                        .with_columns(pl.col("rare_empty").fill_null(0)))
            atomic(small, kept.sort("i", "j").write_parquet)
            side.write_text(json.dumps(args))
        path = small
    ctx = path.with_name(path.name.replace("_pairs_", "_ctx_"))
    if not ctx.exists():
        atomic(ctx, cand_context(path).write_parquet)
    lf = pl.scan_parquet(path).join(pl.scan_parquet(ctx), on="j")
    if knn:  # how the pair looks to the e5 neighbour search (rank 0-19, cosine); null when j is not in i's top-20
        lf = lf.join(pl.scan_parquet(knn).select("i", "j", "knn_rank", "knn_cos"), on=["i", "j"], how="left")
    if knn2:  # the same for the fine-tuned retriever (knn_r_*; features.py passes knn_* through)
        lf = lf.join(pl.scan_parquet(knn2).select("i", "j", knn_r_rank="knn_rank", knn_r_cos="knn_cos"),
                     on=["i", "j"], how="left")
    return s1p, cp, lf


def rare_empty_pairs(s1p, cp, max_df_cand=10):
    """(i, j): same country, same non-empty core name, the core is unique among that country's S1 and appears on at
    most `max_df_cand` candidates, and the candidate has no address."""
    s1 = pl.scan_parquet(s1p).select("idx", "ctry", "core").filter(pl.col("core") != "")
    cand = pl.scan_parquet(cp).select("idx", "ctry", "core", "business_address").filter(pl.col("core") != "")
    s1 = s1.filter(pl.len().over("ctry", "core") == 1)
    cand = cand.filter((pl.len().over("ctry", "core") <= max_df_cand)
                       & (pl.col("business_address").fill_null("").str.strip_chars() == ""))
    return (s1.join(cand, on=["ctry", "core"]).select(pl.col("idx").alias("i"), pl.col("idx_right").alias("j"))
            .collect().unique())


def tfidf(a):
    """Char-3-gram TF-IDF vectorisers for core name and address, fitted once on TRAINING text only, or with
    --tfidf-all on train + test text (label-free; French trigrams, test only, then get real IDF weights)."""
    splits = ("train", "test") if getattr(a, "tfidf_all", False) else ("train",)
    path = a.work / ("tfidf_all.pkl" if len(splits) == 2 else "tfidf.pkl")
    if not path.exists():
        vecs = {}
        for col in ("core", "addr"):
            text = []
            for sp in splits:
                s1p, cp = ensure(a.data, a.work, sp, "s1"), ensure(a.data, a.work, sp, "cand")
                c = pl.scan_parquet(cp).select(col).collect()[col]
                text += [pl.scan_parquet(s1p).select(col).collect()[col], c.sample(min(2_000_000, len(c)), seed=0)]
            text = pl.concat(text)
            vecs[col] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                                        dtype=np.float32, min_df=2).fit(text.to_list())
        atomic(path, lambda t: t.write_bytes(pickle.dumps(vecs)))
    return pickle.loads(path.read_bytes())


def with_emb(a, p, split):
    """Join GPU outputs from --emb-dir: {split}_emb.parquet (embed.py) and {split}_ce.parquet (ce.py), if present."""
    for name in ("emb", "ce") if a.emb_dir else ():
        f = a.emb_dir / f"{split}_{name}.parquet"
        if f.exists():  # read only the rows for p's pairs (the file may cover a larger candidate set)
            e = pl.scan_parquet(f).join(p.select("i", "j").lazy(), on=["i", "j"], how="semi").collect()
            missing = p.height - e.height
            assert missing == 0, f"{missing} {split} pairs missing from {f.name}; re-run the GPU step on these pairs"
            p = p.join(e, on=["i", "j"], how="left", maintain_order="left")
    for pat in getattr(a, "extra", None) or []:  # per-split feature files (i, j, tfw_* / f*_ columns), fitted per split
        e = pl.scan_parquet(pat.format(split=split)).join(p.select("i", "j").lazy(), on=["i", "j"], how="semi").collect()
        if p.height != e.height:
            print(f"  warning: {pat.format(split=split)}: {e.height} rows for {p.height} pairs (missing -> NaN, "
                  f"duplicates -> extra rows)", flush=True)
        p = p.join(e, on=["i", "j"], how="left", maintain_order="left")
    return p


def export_pairs(a):
    """Stage 'pairs': build candidates for train and test, and write the pairs embed.py must score."""
    out = a.work / "emb_in"
    out.mkdir(exist_ok=True)
    s1p, cp, lp = pairs(a, "train")
    lp.filter(pl.col("i").is_in(train_sample(a, s1p).implode())).select("i", "j").collect().write_parquet(
        out / "train_pairs.parquet")
    pairs(a, "test")[2].select("i", "j").collect().write_parquet(out / "test_pairs.parquet")
    print(f"[pairs] wrote {out}/train_pairs.parquet and test_pairs.parquet for embed.py", flush=True)


def train_sample(a, s1p):
    """The S1 records used to train/cross-validate the matcher (fixed seed)."""
    ids = ids_map(s1p).select("idx").collect()["idx"]
    if getattr(a, "exclude_s1", None):  # e.g. the cross-encoder's own training S1 (its scores there are optimistic)
        ids = ids.filter(~ids.is_in(pl.read_parquet(a.exclude_s1, columns=["i"])["i"].unique().implode()))
    return ids.sample(min(a.n_train, len(ids)), seed=0).sort().rename("i")


def ids_map(glob):
    return pl.scan_parquet(glob).select("idx", "entity_id")


def decide(p, prob, t, exclusive):
    q = p.select("i", "j").with_columns(prob=pl.Series(prob))
    if exclusive:  # each S2/S3 record goes to at most one S1 (Source 1 is deduplicated)
        q = q.filter(pl.col("prob") == pl.col("prob").max().over("j"))
    return q.filter(pl.col("prob") >= t).select("i", "j")


def f05(pred, truth, ids):
    """Macro F0.5 over S1 rows `ids` (singletons included). pred/truth: DataFrame(i, j)."""
    per = (ids.to_frame("i")
           .join(pred.group_by("i").len("npred"), on="i", how="left")
           .join(truth.group_by("i").len("ntrue"), on="i", how="left")
           .join(pred.join(truth, on=["i", "j"]).group_by("i").len("tp"), on="i", how="left")
           .fill_null(0))
    P, R = pl.col("tp") / pl.col("npred"), pl.col("tp") / pl.col("ntrue")
    score = (pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0)
             .when(pl.col("tp") == 0).then(0.0)
             .otherwise(1.25 * P * R / (0.25 * P + R)))
    return per.select(score.mean()).item()


def write(path, col, s1p, cp, q):
    lists = (q.lazy().join(ids_map(cp).rename({"idx": "j", "entity_id": "cid"}), on="j")
             .group_by("i").agg(pl.col("cid").unique(maintain_order=True).str.join(",").alias(col)))
    out = (ids_map(s1p).rename({"idx": "i", "entity_id": "source1_entity_id"}).join(lists, on="i", how="left")
           .sort("i").select("source1_entity_id", pl.col(col).fill_null("")).collect())
    out.write_csv(path, separator="\t", quote_style="never")


def truth_pairs(a, s1p, cp):
    gt = pl.scan_csv(a.data / "train" / "train_ground_truth.tsv", separator="\t", infer_schema=False)
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .join(ids_map(s1p), left_on="source1_entity_id", right_on="entity_id")
            .join(ids_map(cp).rename({"idx": "j"}), left_on="matched_entity_ids", right_on="entity_id")
            .select(i="idx", j="j").collect())


def chunks(p):
    """~CHUNK-row slices of p (sorted by i) that never split one S1 record's candidates."""
    i, lo = p["i"].to_numpy(), 0
    while lo < len(i):
        hi = min(lo + CHUNK, len(i))
        while hi < len(i) and i[hi] == i[hi - 1]:
            hi += 1
        yield p.slice(lo, hi - lo)
        lo = hi


def featurise(p, s1p, cp, vecs):
    return pl.concat([features(c, s1p, cp, vecs) for c in chunks(p)])


def train(a):
    s1p, cp, lp = pairs(a, "train")
    ids = train_sample(a, s1p)
    truth = truth_pairs(a, s1p, cp).filter(pl.col("i").is_in(ids.implode()))
    p = with_emb(a, lp.filter(pl.col("i").is_in(ids.implode())).collect().sort("i", "brank"), "train")
    y = p.join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["i", "j"], how="left",
               maintain_order="left")["y"].fill_null(0).to_numpy()
    print(f"[train] {len(ids)} S1 sampled, {p.height} pairs, blocking recall {y.sum() / truth.height:.4f}", flush=True)
    X = featurise(p, s1p, cp, tfidf(a)).to_pandas()
    if a.save_features or a.features_only:  # for offline tuning / retraining with extra columns (retrain_extra.py)
        (pl.from_pandas(X).with_columns(i=p["i"], j=p["j"], y=pl.Series(y, dtype=pl.Int8))
         .write_parquet(a.work / "train_features.parquet"))
        (ids.to_frame().join(truth.group_by("i").len("ntrue"), on="i", how="left").fill_null(0)
         .write_parquet(a.work / "tune_meta.parquet"))
    if a.features_only:  # CPU part only: GPU scores and other columns are joined later
        print(f"[train] features only: wrote {a.work / 'train_features.parquet'}", flush=True)
        return

    oof = np.zeros(p.height)
    for f, (tr, va) in enumerate(GroupKFold(a.folds).split(X, y, p["i"].to_numpy())):
        oof[va] = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        fi = p[va]["i"].unique()
        print(f"  fold {f}: F0.5@0.5 = {f05(decide(p[va], oof[va], 0.5, True), truth.filter(pl.col('i').is_in(fi.implode())), fi):.4f}", flush=True)
    p.select("i", "j").with_columns(y=pl.Series(y), oof=pl.Series(oof, dtype=pl.Float32)).write_parquet(a.work / "oof.parquet")
    best = max((f05(decide(p, oof, t, ex), truth, ids), round(float(t), 2), ex)
               for ex in (True, False) for t in np.arange(0.2, 0.96, 0.02))
    print(f"  CV macro F0.5 = {best[0]:.4f} @ t={best[1]} exclusive={best[2]}", flush=True)

    model = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    model.booster_.save_model(str(a.work / "model.txt"))
    imp = sorted(zip(model.booster_.feature_importance("gain"), X.columns), reverse=True)
    report = {"cv_f05": best[0], "threshold": best[1], "exclusive": best[2], "k": a.k, "n_train_s1": len(ids),
              "train_pairs": p.height, "blocking_recall_train": float(y.sum() / truth.height),
              "top_features": [c for _, c in imp[:15]]}
    (a.work / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def predict(a):
    if not a.features_only:
        rep = json.loads((a.work / "report.json").read_text())
        booster = lgb.Booster(model_file=str(a.work / "model.txt"))
    s1p, cp, lp = pairs(a, "test")
    p = with_emb(a, lp.collect().sort("i", "brank"), "test")
    print(f"[test] {p.height} pairs", flush=True)
    vecs = tfidf(a)
    feat_dir = a.work / "test_features"
    if a.save_test_features or a.features_only:  # kept so test features aren't recomputed (retrain_extra.py)
        feat_dir.mkdir(exist_ok=True)
        for old in feat_dir.glob("part-*.parquet"):  # parts of an earlier (larger) run would be scored too
            old.unlink()
    probs = []
    for n, c in enumerate(chunks(p)):
        X = features(c, s1p, cp, vecs)
        if a.features_only:
            X.with_columns(i=c["i"], j=c["j"]).write_parquet(feat_dir / f"part-{n:05d}.parquet")
            continue
        probs.append(booster.predict(X.to_pandas()))
        if a.save_test_features:
            X.with_columns(i=c["i"], j=c["j"], prob=pl.Series(probs[-1], dtype=pl.Float32)).write_parquet(
                feat_dir / f"part-{n:05d}.parquet")
    if a.features_only:
        print(f"[test] features only: wrote {feat_dir}", flush=True)
        return
    prob = np.concatenate(probs)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.work / "test_prob.npy", prob)
    p.select("i", "j").with_columns(prob=pl.Series(prob, dtype=pl.Float32)).write_parquet(a.work / "test_prob.parquet")
    write(a.out / "candidate_pairs.tsv", "candidate_entity_ids", s1p, cp, p.select("i", "j"))
    q = decide(p, prob, rep["threshold"], rep["exclusive"])
    write(a.out / "matching_results.tsv", "matched_entity_ids", s1p, cp, q)
    print(f"[test] wrote {a.out}: {q.height} matches, {q['i'].n_unique()} S1 with >=1 match", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pairs", "train", "predict", "all"])
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--k", type=int, default=30, help="max candidates per S1 from blocking")
    ap.add_argument("--k-use", type=int, default=0, help="matcher uses only each S1's top-k_use candidates (0 = all k)")
    ap.add_argument("--pool", type=int, default=600, help="blocking pool per S1 before the fuzzy re-rank")
    ap.add_argument("--addr-keys", action="store_true", help="blocking: address word-pair keys (a:)")
    ap.add_argument("--addr-partial", action="store_true", help="blocking re-rank: + address partial ratio")
    ap.add_argument("--block-sample", action="store_true", help="block only the training sample's S1 (CV A/B)")
    ap.add_argument("--n-train", type=int, default=200_000, help="S1 records sampled for model training")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save-features", action="store_true", help="write work/train_features.parquet")
    ap.add_argument("--params", type=Path, help="best_params.json from tune.py (overrides PARAMS)")
    ap.add_argument("--save-test-features", action="store_true",
                    help="predict: also write work/test_features/ (features + i, j, prob) for pseudo-labelling")
    ap.add_argument("--emb-dir", type=Path, help="folder with {train,test}_emb.parquet from embed.py (GPU)")
    ap.add_argument("--features-only", action="store_true",
                    help="train/predict: only compute and save features (no model); GPU/extra columns joined later")
    ap.add_argument("--rescue", action="store_true",
                    help="bidirectional rescue: also keep each candidate's best S1 by re-rank score (beyond --k-use)")
    ap.add_argument("--knn", type=Path, help="dir with {split}_knn.parquet (e5 neighbour search): add top --knn-top "
                                              "pairs as candidates and knn_rank/knn_cos as features")
    ap.add_argument("--knn-top", type=int, default=5)
    ap.add_argument("--knn2", type=Path, help="dir with {split}_knn.parquet from the fine-tuned retriever: + its top "
                                               "--knn2-top pairs as candidates, knn_r_rank/knn_r_cos as features")
    ap.add_argument("--knn2-top", type=int, default=10)
    ap.add_argument("--rare-empty", action="store_true",
                    help="+ candidates: unique S1 core name = rare candidate core with an empty address (+ rare_empty flag)")
    ap.add_argument("--tfidf-all", action="store_true",
                    help="fit the char TF-IDF on train + test text (tfidf_all.pkl); train and predict must both use it")
    ap.add_argument("--extra", action="append", help="per-split feature file with {split} (i, j + tfw_*/f*_ columns)")
    ap.add_argument("--exclude-s1", type=Path, help="parquet with column i: S1 records to keep out of training "
                                                     "(use ce_train.parquet when the cross-encoder feature is on)")
    a = ap.parse_args()
    if (a.rescue or a.knn or a.rare_empty) and not (a.k_use and a.k_use < a.k):
        ap.error("--rescue / --knn / --rare-empty only act together with --k-use < --k (e.g. --k-use 20)")
    if a.params:
        t = json.loads(a.params.read_text())
        PARAMS.update(learning_rate=t["learning_rate"], num_leaves=t["num_leaves"],
                      min_child_samples=t["min_child_samples"], colsample_bytree=t["feature_fraction"],
                      subsample=t["bagging_fraction"], reg_alpha=t["lambda_l1"], reg_lambda=t["lambda_l2"],
                      scale_pos_weight=t["scale_pos_weight"], n_estimators=t["n_estimators"])
        print("using tuned params:", PARAMS, flush=True)
    if a.stage == "pairs":
        export_pairs(a)
    if a.stage in ("train", "all"):
        train(a)
    if a.stage in ("predict", "all"):
        predict(a)
