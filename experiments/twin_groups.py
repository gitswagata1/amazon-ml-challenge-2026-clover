# Step 2 (after swap_asymmetry.py): twin record / S1-sibling kinds of same-address swaps -> work9/twin_{split}.parquet (used by probe1_sim.py)
# Noise or sibling? For a same-address single-word substitution candidate j of S1 i:
#  twin    = another S2/S3 record with j's exact core at the same house-number key (a separate entity has ~3.5 copies)
#  s1_sib  = an S1 record with j's exact core at the same house-number key (a known co-located sibling)
import polars as pl
from rapidfuzz import fuzz, process
for split in ("train", "test"):
    s = pl.read_parquet(f"work9/subs_{split}.parquet")
    s1 = pl.scan_parquet(f"work9/{split}_s1/*.parquet").select(i="idx", a1="addr", h1="hn", core1="core").collect()
    cd = pl.scan_parquet(f"work9/{split}_cand/*.parquet").select(j="idx", c2="country", a2="addr", h2="hn", core2="core").collect()
    twin = cd.filter(pl.col("h2") != "").group_by("c2", "core2", "h2").len("nrec")
    s1k = pl.scan_parquet(f"work9/{split}_s1/*.parquet").select(c2="country", core2="core", h2="hn").collect().filter(pl.col("h2") != "").group_by("c2", "core2", "h2").len("ns1")
    d = s.join(s1, on="i").join(cd, on="j").filter(pl.col("h2") != "")
    asim = process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    d = d.with_columns(asim=pl.Series(asim)).filter(pl.col("asim") >= 95)
    d = d.join(twin, on=["c2", "core2", "h2"], how="left").join(s1k, on=["c2", "core2", "h2"], how="left").fill_null(0)
    d = d.with_columns(kind=pl.when(pl.col("ns1") > 0).then(pl.lit("S1 sibling exists")).when(pl.col("nrec") > 1).then(pl.lit("twin record")).otherwise(pl.lit("single record")),
                       noise=pl.col("lr_add") > 2)
    aggs = [pl.len().alias("n"), pl.col("p").mean().round(3).alias("p"), (pl.col("p") >= 0.74).mean().round(3).alias("acc")]
    if "y" in d.columns: aggs.append(pl.col("y").mean().round(3).alias("pos"))
    with pl.Config(tbl_rows=40):
        print(split); print(d.group_by("c", "noise", "kind").agg(aggs).sort("c", "noise", "kind"))
    d.write_parquet(f"work9/twin_{split}.parquet")
