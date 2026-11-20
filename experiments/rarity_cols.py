"""Name-rarity features (laptop's idea 1b, A/B +0.0038) for a split's final pairs -> <work>/rarity_{split}.parquet,
used by run.py --extra. Label-free counts over ALL records of that split (so test/France counts come from test):
  name_df_s1          #S1 with the same (ctry, core)                      (on i)
  name_df_cand        #candidates (S2+S3, any address) with the same (ctry, core)  (on j)
  cand_addr_empty     candidate address empty after strip                 (on j)
  name_df_s1_x_empty  name_df_s1 * cand_addr_empty
  name_df_s1_eq1, name_df_cand_le1: scale-free flags (raw counts shift with split size)
Empty core -> counts null.

python experiments/rarity_cols.py --work work9 --tag k30_p1500_ap_ak_use20_rs_knn5_re
"""
import argparse
from pathlib import Path

import polars as pl


def rarity(work: Path, split: str, pairs: Path) -> pl.DataFrame:
    s1 = pl.scan_parquet(str(work / f"{split}_s1" / "*.parquet")).select("idx", "ctry", "core")
    cand = pl.scan_parquet(str(work / f"{split}_cand" / "*.parquet")).select("idx", "ctry", "core", "business_address")
    df = lambda lf: pl.when(pl.col("core") != "").then(pl.len().over("ctry", "core")).cast(pl.Float32)
    a = s1.select(i="idx", name_df_s1=df(s1))
    b = cand.select(j="idx", name_df_cand=df(cand),
                    cand_addr_empty=(pl.col("business_address").fill_null("").str.strip_chars() == "").cast(pl.Int8))
    return (pl.scan_parquet(pairs).select("i", "j").join(a, on="i", how="left").join(b, on="j", how="left")
            .with_columns(name_df_s1_x_empty=pl.col("name_df_s1") * pl.col("cand_addr_empty"),
                          name_df_s1_eq1=(pl.col("name_df_s1") == 1).cast(pl.Int8),
                          name_df_cand_le1=(pl.col("name_df_cand") <= 1).cast(pl.Int8))
            .collect())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work9"))
    ap.add_argument("--tag", default="k30_p1500_ap_ak_use20_rs_knn5_re")
    a = ap.parse_args()
    for split in ("train", "test"):
        r = rarity(a.work, split, a.work / f"{split}_pairs_{a.tag}.parquet")
        r.write_parquet(a.work / f"rarity_{split}.parquet")
        print(f"{split}: {r.height} pairs, unique-S1-name share {r['name_df_s1_eq1'].mean():.3f}, "
              f"empty-address candidates {r['cand_addr_empty'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
