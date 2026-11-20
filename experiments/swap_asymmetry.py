# Step 1 of the France analysis (27 Sep): single-word swaps per split + label-free add/remove asymmetry -> work9/subs_{split}.parquet, asym_{split}.parquet
# Label-free noise-word detector: in single-token substitutions (candidate core = S1 core with ONE word replaced, not a typo),
# the generator's noise words are added far more often than removed; sibling words are symmetric.
# Check on train (labels) that the asymmetry separates matches from non-matches, then look at France.
import polars as pl, numpy as np
from rapidfuzz import fuzz, process
def subs(split, f, col):
    p = pl.read_parquet(f).rename({col: "p"})
    s1 = pl.scan_parquet(f"work9/{split}_s1/*.parquet").select(i="idx", c="country", n1="core").collect()
    cd = pl.scan_parquet(f"work9/{split}_cand/*.parquet").select(j="idx", n2="core").collect()
    d = p.join(s1, on="i").join(cd, on="j").with_columns(t1=pl.col("n1").str.split(" ").list.unique(), t2=pl.col("n2").str.split(" ").list.unique())
    d = d.with_columns(ra=pl.col("t1").list.set_difference("t2"), rb=pl.col("t2").list.set_difference("t1"), k=pl.col("t1").list.set_intersection("t2").list.len())
    d = d.filter((pl.col("ra").list.len() == 1) & (pl.col("rb").list.len() == 1) & (pl.col("k") >= 1)).with_columns(
        rem=pl.col("ra").list.first(), add=pl.col("rb").list.first()).drop("t1", "t2", "ra", "rb", "n1", "n2")
    sim = process.cpdist(d["rem"].to_list(), d["add"].to_list(), scorer=fuzz.ratio, workers=-1)
    return d.with_columns(tsim=pl.Series(sim)).filter(pl.col("tsim") < 70)
def asym(d):
    a = d.group_by("c", w="add").len("n_add")
    r = d.group_by("c", w="rem").len("n_rem")
    m = a.join(r, on=["c", "w"], how="full", coalesce=True).fill_null(0)
    return m.with_columns(lr=((pl.col("n_add") + 5) / (pl.col("n_rem") + 5)).log())
tr = subs("train", "work9/oof_v9.parquet", "oof")
m = asym(tr)
tr = tr.join(m.select("c", add="w", lr_add="lr", n_add="n_add"), on=["c", "add"], how="left")
tr = tr.with_columns(b=pl.col("lr_add").cut([0, 1, 2, 3]))
print("train single substitutions:", tr.height)
print(tr.group_by("c", "b").agg(n=pl.len(), pos=pl.col("y").mean().round(3), p=pl.col("p").mean().round(3), acc=(pl.col("p") >= 0.74).mean().round(3)).sort("c", "b"))
print(m.filter(pl.col("lr") > 2).sort("n_add", descending=True).head(12))
te = subs("test", "work9/test_prob_v9.parquet", "prob")
mt = asym(te)
te = te.join(mt.select("c", add="w", lr_add="lr", n_add="n_add"), on=["c", "add"], how="left").with_columns(b=pl.col("lr_add").cut([0, 1, 2, 3]))
print(te.group_by("c", "b").agg(n=pl.len(), p=pl.col("p").mean().round(3), acc=(pl.col("p") >= 0.74).mean().round(3)).sort("c", "b"))
print(mt.filter((pl.col("c") == "France") & (pl.col("lr") > 1)).sort("n_add", descending=True).head(15))
tr.write_parquet("work9/subs_train.parquet"); te.write_parquet("work9/subs_test.parquet")
m.write_parquet("work9/asym_train.parquet"); mt.write_parquet("work9/asym_test.parquet")
