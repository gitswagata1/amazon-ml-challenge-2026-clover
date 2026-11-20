"""Re-sort test: which cheap score best lifts true matches from the blocking pool into each S1's top-20?

Uses the pool saved by experiments/recall.py (same 20k S1 sample, key cap 10k). For every pooled pair it
computes a few cheap rapidfuzz scores, then ranks each S1's pool by: the current blocking._rescore formula,
hand-weighted variants, and a small LightGBM ranker (2-fold by S1, so every S1 is scored out-of-fold).
Reports top-20/30 recall and oracle macro F0.5 (perfect matcher on the top-20) at pool 600 and 1500.
Also sizes the embedding-search ceiling: true pairs sharing no blocking key at all, with examples.

python experiments/resort.py --work work --data dataset --pool work/recall_diag/pool.parquet
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import _keys  # noqa: E402
from recall import sample_truth  # noqa: E402

POOLS = (600, 1500)
COLS = ["idx", "core", "sk", "addr", "hn", "bk", "ctry"]
SCORES = {  # name -> (text column, scorer)
    "sk_tset": ("sk", fuzz.token_set_ratio), "addr_tset": ("addr", fuzz.token_set_ratio),
    "core_tset": ("core", fuzz.token_set_ratio), "core_ratio": ("core", fuzz.ratio),
    "core_partial": ("core", fuzz.partial_ratio), "core_jw": ("core", JaroWinkler.normalized_similarity),
    "addr_partial": ("addr", fuzz.partial_ratio), "addr_tsort": ("addr", fuzz.token_sort_ratio),
}


def cheap_scores(p, s1c, cc, workers, step=2_000_000):
    out = []
    for lo in range(0, p.height, step):
        q = p.slice(lo, step)
        a = q.select("i").join(s1c, left_on="i", right_on="idx", how="left", maintain_order="left")
        b = q.select("j").join(cc, left_on="j", right_on="idx", how="left", maintain_order="left")
        f = {k: process.cpdist(a[c].to_list(), b[c].to_list(), scorer=s, workers=workers).astype(np.float32)
             for k, (c, s) in SCORES.items()}
        f["has_addr"] = ((a["addr"] != "") & (b["addr"] != "")).to_numpy()
        f["hn"] = np.array([bool(set(x.split(" | ")) & set(y.split(" | "))) and x != ""
                            for x, y in zip(a["hn"].to_list(), b["hn"].to_list())])
        out.append(q.with_columns(**{k: pl.Series(v) for k, v in f.items()}))
        print(f"  scored {lo + q.height}/{p.height}", flush=True)
    return pl.concat(out)


def evaluate(p, score, truth, n_s1, pool):
    """top-k recall and oracle macro F0.5 when each S1's pool (brank <= pool) is ranked by `score`."""
    r = (p.filter(pl.col("brank") <= pool).select("i", "j", s=score)
         .with_columns(rrank=pl.col("s").rank("ordinal", descending=True).over("i")))
    hit = truth.join(r.select("i", "j", "rrank"), on=["i", "j"], how="left")  # null rrank = not in pool = miss
    hit = hit.with_columns(pl.col("rrank").fill_null(2**31))
    per = hit.group_by("i").agg(R=(pl.col("rrank") <= 20).sum() / pl.len())
    f = per.select(pl.when(pl.col("R") == 0).then(0.0).otherwise(1.25 * pl.col("R") / (0.25 + pl.col("R"))))
    return dict(top20=(hit["rrank"] <= 20).mean(), top30=(hit["rrank"] <= 30).mean(),
                in_pool=(hit["rrank"] < 2**31).mean(),
                oracle_f05=(f.to_series().sum() + n_s1 - per.height) / n_s1)


def no_key_pairs(truth, s1p, cp):
    """True pairs whose records share no blocking key at all (ignoring the df cap): only embeddings can reach them."""
    a = pl.scan_parquet(s1p).filter(pl.col("idx").is_in(truth["i"].unique().implode())).select(COLS).collect()
    b = pl.scan_parquet(cp).filter(pl.col("idx").is_in(truth["j"].unique().implode())).select(COLS).collect()
    ka, kb = _keys(a.rename({"idx": "i"}), "i"), _keys(b.rename({"idx": "j"}), "j")
    shared = truth.join(ka, on="i").join(kb, on=["j", "h"], how="semi").select("i", "j").unique()
    return truth.join(shared, on=["i", "j"], how="anti"), a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--pool", type=Path, default=Path("work/recall_diag/pool.parquet"))
    ap.add_argument("--n", type=int, default=20_000, help="must match the recall.py run that wrote --pool")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    s1p, cp, _, sample, truth = sample_truth(a.work, a.data, a.n)
    truth = truth.select("i", "j")
    n_s1 = len(sample)

    p = (pl.scan_parquet(a.pool).filter((pl.col("cap") == 10_000) & (pl.col("brank") <= max(POOLS)))
         .select("i", "j", "bscore", "brank").collect().sort("i", "brank"))
    s1c = pl.scan_parquet(s1p).filter(pl.col("idx").is_in(p["i"].unique().implode())).select(COLS[:5]).collect()
    cc = pl.scan_parquet(cp).filter(pl.col("idx").is_in(p["j"].unique().implode())).select(COLS[:5]).collect()
    print(f"[resort] {p.height} pooled pairs, {n_s1} S1, {truth.height} true pairs", flush=True)
    p = cheap_scores(p, s1c, cc, a.workers)
    del s1c, cc
    p = (p.with_columns(b_rel=pl.col("bscore") / pl.col("bscore").max().over("i"))
         .join(truth.with_columns(y=pl.lit(True)), on=["i", "j"], how="left", maintain_order="left")
         .with_columns(pl.col("y").fill_null(False)))

    addr = lambda c: pl.when(pl.col("has_addr")).then(pl.col(c)).otherwise(50)  # noqa: E731
    base = pl.col("sk_tset") + addr("addr_tset") + 40 * pl.col("hn") + 2 * pl.col("bscore")
    variants = {
        "current (blocking._rescore)": base,
        "+ addr_partial": base + addr("addr_partial"),
        "+ core_tset": base + pl.col("core_tset"),
        "+ core_jw*100": base + 100 * pl.col("core_jw"),
        "+ core_tset + addr_partial": base + pl.col("core_tset") + addr("addr_partial"),
        "+ core_tset + addr_partial + core_partial": base + pl.col("core_tset") + addr("addr_partial")
                                                     + pl.col("core_partial"),
        "current, bscore weight 4": base + 2 * pl.col("bscore"),
        "current, bscore weight 1": base - pl.col("bscore"),
    }

    # small learned ranker on the same cheap scores, out-of-fold by S1 (2 folds)
    feats = list(SCORES) + ["has_addr", "hn", "bscore", "brank", "b_rel"]
    fold = (p["i"].hash(7) % 2).to_numpy()
    X, y = p.select(feats).to_numpy().astype(np.float32), p["y"].to_numpy()
    prob = np.zeros(p.height, np.float32)
    for k in (0, 1):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, min_child_samples=100,
                               subsample=0.5, subsample_freq=1, verbose=-1, n_jobs=a.workers)
        m.fit(X[fold != k], y[fold != k])
        prob[fold == k] = m.predict_proba(X[fold == k])[:, 1]
    p = p.with_columns(lgbm=pl.Series(prob))
    variants["small LightGBM on cheap scores (out-of-fold)"] = pl.col("lgbm")
    imp = sorted(zip(m.booster_.feature_importance("gain"), feats), reverse=True)
    print("  lgbm gain importance:", ", ".join(f"{f} {g / sum(x for x, _ in imp):.2f}" for g, f in imp[:8]))

    rows = [dict(variant=v, pool=pool, **evaluate(p, e, truth, n_s1, pool)) for v, e in variants.items() for pool in POOLS]
    res = pl.DataFrame(rows)
    res.write_csv(a.pool.with_name("resort.tsv"), separator="\t")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, fmt_str_lengths=50, tbl_width_chars=200):
        print(res.sort("pool", "top20"))

    # embedding-search ceiling: add every no-key true pair to the top-20 of the current setting
    nk, _, _ = no_key_pairs(truth, s1p, cp)
    cur = (p.filter(pl.col("brank") <= 600).select("i", "j", s=base)
           .with_columns(rrank=pl.col("s").rank("ordinal", descending=True).over("i")).filter(pl.col("rrank") <= 20))
    hit = truth.join(pl.concat([cur.select("i", "j"), nk]).unique().with_columns(h=pl.lit(1)), on=["i", "j"], how="left")
    per = hit.group_by("i").agg(R=pl.col("h").is_not_null().sum() / pl.len())
    f = per.select(pl.when(pl.col("R") == 0).then(0.0).otherwise(1.25 * pl.col("R") / (0.25 + pl.col("R"))))
    print(f"\n[embedding ceiling] {nk.height} true pairs ({nk.height / truth.height:.4f}) share no key; "
          f"{nk['i'].n_unique()} S1 affected ({nk['i'].n_unique() / n_s1:.4f}). Oracle F0.5 if ALL were found: "
          f"{(f.to_series().sum() + n_s1 - per.height) / n_s1:.4f}")
    show = ["idx", "business_name", "business_address"]
    ex = nk.sample(min(25, nk.height), seed=0)
    txt = lambda glob, ids, sfx: (pl.scan_parquet(glob).filter(pl.col("idx").is_in(ids.implode()))  # noqa: E731
                                  .select(show).collect().rename(lambda c: c + sfx))
    ex = (ex.join(txt(s1p, ex["i"], "_1"), left_on="i", right_on="idx_1")
          .join(txt(cp, ex["j"], "_2"), left_on="j", right_on="idx_2"))
    for r in ex.iter_rows(named=True):
        print(f"  S1 {r['business_name_1']!r:45.45} | {r['business_address_1']!r:50.50}\n"
              f"  S* {r['business_name_2']!r:45.45} | {r['business_address_2']!r:50.50}\n")


if __name__ == "__main__":
    main()
