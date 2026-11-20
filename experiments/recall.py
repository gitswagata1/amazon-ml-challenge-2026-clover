"""Where are true matches lost? Blocking-recall diagnostic on a sample of training S1 records.

Re-runs the blocking of blocking.block() for several key caps (max_df) and pool sizes, and reports for every
setting the share of true pairs that (a) reach the pool, (b) survive the fuzzy re-rank into the top-20/30,
plus the oracle macro F0.5 (perfect matcher on the top-20). Also buckets the pairs that no setting reaches:
cross-country, no shared key at all (only embedding search could find them), or shared keys only above the cap.
Reads the normalised caches read-only; writes nothing outside --out.

python experiments/recall.py --work work --data dataset --n 50000 --out work/recall_diag
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from blocking import keys  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

CAPS = (10_000, 30_000, 50_000)
POOLS = (600, 1000, 1500, 2000)
KS = (20, 30)


def text_score(q, s1c, cc, workers):
    """The text part of blocking._rescore (name + address token-set, +40 house-number overlap); bscore added later."""
    a = q.select("i").join(s1c, left_on="i", right_on="idx", how="left", maintain_order="left")
    b = q.select("j").join(cc, left_on="j", right_on="idx", how="left", maintain_order="left")
    name = process.cpdist(a["sk"].to_list(), b["sk"].to_list(), scorer=fuzz.token_set_ratio, workers=workers)
    addr = process.cpdist(a["addr"].to_list(), b["addr"].to_list(), scorer=fuzz.token_set_ratio, workers=workers)
    has = ((a["addr"] != "") & (b["addr"] != "")).to_numpy()
    hn = np.array([bool(set(x.split(" | ")) & set(y.split(" | "))) and x != ""
                   for x, y in zip(a["hn"].to_list(), b["hn"].to_list())])
    return name + np.where(has, addr, 50) + 40 * hn


def country(c, s1p, cp, sample, truth, chunk, workers):
    """-> (pool pairs: cap, i, j, bscore, brank, text; truth pairs of this country with min shared-key df)."""
    ck = keys(cp, "j", c)
    df = ck.group_by("h").len("df")
    n = df["df"].sum()
    df = df.with_columns(w=(np.log(n) - pl.col("df").log()).cast(pl.Float32))
    sk = keys(s1p, "i", c).filter(pl.col("i").is_in(sample.implode()))
    # min document frequency over the keys each true pair shares (null = shares no key at all)
    t = truth.filter(pl.col("i").is_in(sk["i"].unique().implode()))
    tk = (t.join(sk, on="i").join(ck, on=["j", "h"], how="semi").join(df.select("h", "df"), on="h")
          .group_by("i", "j").agg(min_df=pl.col("df").min()))
    t = t.join(tk, on=["i", "j"], how="left")
    ck = ck.join(df.filter(pl.col("df") <= max(CAPS)), on="h")  # (j, h, df, w)
    del df
    ids, out = sk["i"].unique().sort(), []
    for lo in range(0, len(ids), chunk):
        part = ids.slice(lo, chunk)
        q = sk.filter(pl.col("i").is_between(part[0], part[-1])).join(ck, on="h")
        for cap in CAPS:
            out.append(q.filter(pl.col("df") <= cap).group_by("i", "j").agg(bscore=pl.col("w").sum())
                       .sort(["i", "bscore"], descending=[False, True]).group_by("i", maintain_order=True)
                       .head(max(POOLS))
                       .with_columns(cap=pl.lit(cap, pl.UInt32),
                                     brank=pl.int_range(1, pl.len() + 1, dtype=pl.UInt16).over("i")))
    del ck, sk
    pool = pl.concat(out)
    cols = ["idx", "sk", "addr", "hn"]
    uniq = pool.select("i", "j").unique()
    s1c = pl.scan_parquet(s1p).filter(pl.col("idx").is_in(uniq["i"].unique().implode())).select(cols).collect()
    cc = pl.scan_parquet(cp).filter(pl.col("idx").is_in(uniq["j"].unique().implode())).select(cols).collect()
    uniq = uniq.with_columns(text=pl.Series(text_score(uniq, s1c, cc, workers), dtype=pl.Float32))
    print(f"  country={c!r}: {len(ids)} S1, {pool.height} pooled pairs, {uniq.height} scored", flush=True)
    return pool.join(uniq, on=["i", "j"]), t


def sample_truth(work: Path, data: Path, n: int):
    """-> (s1 glob, cand glob, s1 frame, sampled S1 idx (seed 1), true pairs (i, j, cctry) of the sample)."""
    s1p, cp = str(work / "train_s1/*.parquet"), str(work / "train_cand/*.parquet")
    s1 = pl.scan_parquet(s1p).select("idx", "entity_id", "ctry").collect()
    sample = s1["idx"].sample(n, seed=1).sort()
    gt = pl.scan_csv(data / "train/train_ground_truth.tsv", separator="\t", infer_schema=False)
    truth = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "")
             .join(s1.lazy().select("idx", "entity_id"), left_on="source1_entity_id", right_on="entity_id")
             .join(pl.scan_parquet(cp).select(j="idx", entity_id="entity_id", cctry="ctry"),
                   left_on="matched_entity_ids", right_on="entity_id")
             .select(i="idx", j="j", cctry="cctry").collect()
             .filter(pl.col("i").is_in(sample.implode())))
    return s1p, cp, s1, sample, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("work/recall_diag"))
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s1p, cp, s1, sample, truth = sample_truth(a.work, a.data, a.n)
    print(f"[recall] {len(sample)} S1 sampled, {truth.height} true pairs, "
          f"{truth['i'].n_unique()} S1 with matches", flush=True)

    pools, tlist = [], []
    for c in s1.filter(pl.col("idx").is_in(sample.implode()))["ctry"].unique().sort():
        p, t = country(c, s1p, cp, sample, truth.filter(pl.col("cctry") == c).select("i", "j"), a.chunk, a.workers)
        pools.append(p)
        tlist.append(t)
    pool = pl.concat(pools).with_columns(rscore=pl.col("text") + 2 * pl.col("bscore"))
    pool.write_parquet(a.out / "pool.parquet")
    tr = pl.concat(tlist)
    ntrue = truth.height

    rows = []
    for cap in CAPS:
        for pl_size in POOLS:
            ranked = (pool.filter((pl.col("cap") == cap) & (pl.col("brank") <= pl_size))
                      .sort(["i", "rscore"], descending=[False, True])
                      .with_columns(rrank=pl.int_range(1, pl.len() + 1).over("i")))
            hit = truth.select("i", "j").join(ranked.select("i", "j", "rrank"), on=["i", "j"], how="left")
            row = dict(cap=cap, pool=pl_size, in_pool=hit["rrank"].is_not_null().sum() / ntrue)
            for k in KS:
                row[f"top{k}"] = (hit["rrank"] <= k).sum() / ntrue
            # oracle macro F0.5 over all sampled S1 (perfect matcher on the top-20; singletons score 1)
            per = hit.group_by("i").agg(R=(pl.col("rrank") <= 20).sum() / pl.len())
            f = per.select(pl.when(pl.col("R") == 0).then(0.0).otherwise(1.25 * pl.col("R") / (0.25 + pl.col("R"))))
            row["oracle_f05_top20"] = (f.to_series().sum() + (len(sample) - per.height)) / len(sample)
            rows.append(row)
    res = pl.DataFrame(rows)
    res.write_csv(a.out / "recall.tsv", separator="\t")

    cross = ntrue - tr.height
    nokey = tr["min_df"].is_null().sum()
    buckets = {"cross-country / candidate country differs": cross, "no shared key at all": nokey}
    lo = 0
    for cap in CAPS + (10**9,):
        buckets[f"cheapest shared key df in ({lo}, {cap}]"] = tr.filter(
            (pl.col("min_df") > lo) & (pl.col("min_df") <= cap)).height
        lo = cap
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(res)
    for k, v in buckets.items():
        print(f"  {k:45s} {v:8d}  {v / ntrue:.4f}")


if __name__ == "__main__":
    main()
