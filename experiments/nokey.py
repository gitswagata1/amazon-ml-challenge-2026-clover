"""Is embedding search still worth it? Measures what the best blocking (arm C of work/ab7) still misses.

For the 400k training S1 of the ab7 A/B: true pairs missing from the top-20, split into cross-country,
no shared blocking key at all even with address keys (only embedding search could reach these), and
key shared but ranked out. Prints oracle macro F0.5 (perfect matcher) for the top-20 as is and with every
no-key pair added: the ceiling for embedding search.

python experiments/nokey.py --work work/ab7/addr --pairs train_pairs_k30_use20.parquet
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from blocking import _keys  # noqa: E402
from run import train_sample, truth_pairs  # noqa: E402

COLS = ["idx", "ctry", "bk", "sk", "hn", "addr"]


def oracle(truth, found, n_s1):
    per = (truth.join(found.with_columns(h=pl.lit(1)), on=["i", "j"], how="left")
           .group_by("i").agg(R=pl.col("h").is_not_null().sum() / pl.len()))
    f = per.select(pl.when(pl.col("R") == 0).then(0.0).otherwise(1.25 * pl.col("R") / (0.25 + pl.col("R"))))
    return (f.to_series().sum() + n_s1 - per.height) / n_s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work/ab7/addr"))
    ap.add_argument("--pairs", default="train_pairs_k30_use20.parquet")
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--n-train", type=int, default=400_000)
    a = ap.parse_args()
    s1p, cp = str(a.work / "train_s1/*.parquet"), str(a.work / "train_cand/*.parquet")
    ids = train_sample(a, s1p)
    truth = truth_pairs(a, s1p, cp).filter(pl.col("i").is_in(ids.implode()))
    top = pl.read_parquet(a.work / a.pairs, columns=["i", "j"]).filter(pl.col("i").is_in(ids.implode()))
    miss = truth.join(top, on=["i", "j"], how="anti")

    s1 = pl.scan_parquet(s1p).filter(pl.col("idx").is_in(miss["i"].unique().implode())).select(COLS).collect()
    cc = pl.scan_parquet(cp).filter(pl.col("idx").is_in(miss["j"].unique().implode())).select(COLS).collect()
    miss = (miss.join(s1.select(i="idx", c1="ctry"), on="i").join(cc.select(j="idx", c2="ctry"), on="j"))
    ka, kb = _keys(s1.rename({"idx": "i"}), "i", True), _keys(cc.rename({"idx": "j"}), "j", True)
    keyed = miss.select("i", "j").join(ka, on="i").join(kb, on=["j", "h"], how="semi").select("i", "j").unique()
    cross = miss.filter(pl.col("c1") != pl.col("c2")).select("i", "j")
    nokey = miss.filter(pl.col("c1") == pl.col("c2")).select("i", "j").join(keyed, on=["i", "j"], how="anti")

    n, n_s1 = truth.height, len(ids)
    print(f"[nokey] {n_s1} S1, {n} true pairs, top-20 recall {1 - miss.height / n:.4f}")
    for name, d in (("missed in total", miss), ("  cross-country", cross), ("  no shared key (embedding-only)", nokey),
                    ("  key shared, ranked out of top-20", keyed)):
        print(f"  {name:38s} {d.height:8d}  {d.height / n:.4f} of true pairs, {d['i'].n_unique() / n_s1:.4f} of S1")
    base = oracle(truth, top, n_s1)
    print(f"  oracle F0.5 top-20 now: {base:.4f}; + all no-key pairs: {oracle(truth, pl.concat([top, nokey]), n_s1):.4f}"
          f"; + all missed: {oracle(truth, truth, n_s1):.4f}")
    show = ["idx", "business_name", "business_address"]
    ex = nokey.sample(min(15, nokey.height), seed=0)
    ex = (ex.join(pl.scan_parquet(s1p).filter(pl.col("idx").is_in(ex["i"].implode())).select(show).collect()
                  .rename(lambda c: c + "_1"), left_on="i", right_on="idx_1")
          .join(pl.scan_parquet(cp).filter(pl.col("idx").is_in(ex["j"].implode())).select(show).collect()
                .rename(lambda c: c + "_2"), left_on="j", right_on="idx_2"))
    for r in ex.iter_rows(named=True):
        print(f"  S1 {r['business_name_1']!r:45.45} | {r['business_address_1']!r:55.55}\n"
              f"  S* {r['business_name_2']!r:45.45} | {r['business_address_2']!r:55.55}\n")


if __name__ == "__main__":
    main()
