"""Build cross-encoder training sets from training candidate pairs (with labels). Run after `run.py pairs`.

Mode 1 (e5-small `ce`): pairs of 100k S1 OUTSIDE the matcher's training sample, so its scores are never optimistic.
    python ce_data.py --n-train 600000 --out work/ce              # -> ce_train.parquet (i, j, y)
Mode 2 (out-of-fold re-ranker): all training S1 are split into two halves by ce_half(i); the model for half h trains on
a sample of half h and scores only the other half (and test), so every training S1 gets an out-of-fold score.
    python ce_data.py --half 0 --n-ce 200000 --k-use 20 --out work/ce2   # -> ce_train_h0.parquet
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import polars as pl

import run


def ce_half(i: pl.Expr) -> pl.Expr:
    """Fixed 2-way split of training S1 ids (Knuth multiplicative hash); used by every machine."""
    return ((i.cast(pl.UInt64) * 2654435761) % 2**32) % 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--n-train", type=int, default=600_000, help="must match the matcher's --n-train")
    ap.add_argument("--n-ce", type=int, default=100_000, help="S1 records whose pairs train the cross-encoder")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("work/ce"))
    ap.add_argument("--half", type=int, choices=[0, 1], help="out-of-fold mode: train on a sample of this half")
    ap.add_argument("--k-use", type=int, default=0, help="keep only each S1's top-k_use candidates (0 = all k)")
    a = ap.parse_args()
    s1p = f"{a.work}/train_s1/*.parquet"
    if a.half is not None:
        ids = pl.scan_parquet(s1p).select("idx").filter(ce_half(pl.col("idx")) == a.half).collect()["idx"]
        others = ids.sample(min(a.n_ce, len(ids)), seed=11 + a.half)
        truth = run.truth_pairs(SimpleNamespace(data=a.data), s1p, f"{a.work}/train_cand/*.parquet")
        lp = pl.scan_parquet(a.work / f"train_pairs_k{a.k}.parquet")
        if a.k_use:
            lp = lp.filter(pl.col("brank") <= a.k_use)
        tr = (lp.select("i", "j").filter(pl.col("i").is_in(others.implode())).collect()
              .join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["i", "j"], how="left")
              .with_columns(pl.col("y").fill_null(0)))
        a.out.mkdir(parents=True, exist_ok=True)
        tr.write_parquet(a.out / f"ce_train_h{a.half}.parquet")
        print(f"half {a.half}: {tr.height} pairs ({tr['y'].sum()} true) from {tr['i'].n_unique()} S1")
        return
    sample = run.train_sample(SimpleNamespace(n_train=a.n_train, exclude_s1=None), s1p)
    ids = pl.scan_parquet(s1p).select("idx").collect()["idx"]
    others = ids.filter(~ids.is_in(sample.implode())).sample(a.n_ce, seed=11)
    truth = run.truth_pairs(SimpleNamespace(data=a.data), s1p, f"{a.work}/train_cand/*.parquet")
    tr = (pl.scan_parquet(a.work / f"train_pairs_k{a.k}.parquet").select("i", "j")
          .filter(pl.col("i").is_in(others.implode())).collect()
          .join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["i", "j"], how="left")
          .with_columns(pl.col("y").fill_null(0)))
    a.out.mkdir(parents=True, exist_ok=True)
    tr.write_parquet(a.out / "ce_train.parquet")
    print(f"ce_train: {tr.height} pairs ({tr['y'].sum()} true) from {tr['i'].n_unique()} S1; "
          f"overlap with the matcher's sample: {others.is_in(sample.implode()).sum()}")


if __name__ == "__main__":
    main()
