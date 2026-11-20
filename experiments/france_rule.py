"""The France rule adopted from the LB probe (27 Sep, +0.001118): list the test pairs of group G-common, exactly as probed.
G-common = candidate j of S1 i where
  - core(j) = core(i) with ONE word swapped (one word removed, one added, >= 1 shared, the two words not typo-like: ratio < 70),
  - the added word is not a generator noise word (log((n_add+5)/(n_rem+5)) <= 2 over all such swaps in this split+country),
  - both words are common name words (in >= 20 S1 cores of this split+country),
  - same address (token-set ratio >= 95, house-number key present),
  - no other candidate record with j's (core, house key) and no S1 with j's (core, house key).
Output: parquet (source1_entity_id, e2) for France S1; apply with france_probe.py --mode reject.

python experiments/france_rule.py --work work10 --pairs work10/test_prob_v10ce2.parquet --out work10/g_test_common.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="parquet with i, j = all scored test pairs")
    ap.add_argument("--country", default="France", help="country to output, or ALL")
    ap.add_argument("--split", default="test", help="test (default) or train (for simulations)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr-max", type=float, help="max log add/remove ratio of the added word (default: 2 same = as probed, 0.5 different)")
    ap.add_argument("--address", choices=("same", "different"), default="same",
                    help="same = the LB-proven rule; different = P2' (same swap, address differs; twin/S1 checks skipped)")
    a = ap.parse_args()
    if a.lr_max is None:
        a.lr_max = 2.0 if a.address == "same" else 0.5
    s1 = pl.scan_parquet(str(a.work / f"{a.split}_s1" / "*.parquet")).select(i="idx", c="country", n1="core", a1="addr", h1="hn",
                                                                       s="entity_id").collect()
    cd = pl.scan_parquet(str(a.work / f"{a.split}_cand" / "*.parquet")).select(j="idx", c2="country", n2="core", a2="addr", h2="hn",
                                                                         e2="entity_id").collect()
    d = pl.read_parquet(a.pairs, columns=["i", "j"]).join(s1, on="i").join(cd, on="j")
    d = d.with_columns(t1=pl.col("n1").str.split(" ").list.unique(), t2=pl.col("n2").str.split(" ").list.unique())
    d = d.with_columns(ra=pl.col("t1").list.set_difference("t2"), rb=pl.col("t2").list.set_difference("t1"),
                       k=pl.col("t1").list.set_intersection("t2").list.len())
    d = d.filter((pl.col("ra").list.len() == 1) & (pl.col("rb").list.len() == 1) & (pl.col("k") >= 1)).with_columns(
        rem=pl.col("ra").list.first(), add=pl.col("rb").list.first()).drop("t1", "t2", "ra", "rb")
    d = d.with_columns(tsim=pl.Series(process.cpdist(d["rem"].to_list(), d["add"].to_list(), scorer=fuzz.ratio, workers=-1))) \
         .filter(pl.col("tsim") < 70)
    m = (d.group_by("c", w="add").len("n_add").join(d.group_by("c", w="rem").len("n_rem"), on=["c", "w"], how="full", coalesce=True)
         .fill_null(0).with_columns(lr=((pl.col("n_add") + 5) / (pl.col("n_rem") + 5)).log()))
    d = d.join(m.select("c", add="w", lr_add="lr"), on=["c", "add"], how="left").filter(pl.col("lr_add") <= a.lr_max)
    wdf = s1.with_columns(w=pl.col("n1").str.split(" ").list.unique()).explode("w").group_by("c", "w").len("df")
    d = (d.join(wdf.rename({"w": "add", "df": "df_add"}), on=["c", "add"], how="left")
         .join(wdf.rename({"w": "rem", "df": "df_rem"}), on=["c", "rem"], how="left").fill_null(0)
         .filter((pl.col("df_add") >= 20) & (pl.col("df_rem") >= 20)))
    if a.address == "different":
        d = d.with_columns(asim=pl.Series(process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
        d = d.filter((pl.col("a2") != "") & (pl.col("asim") < 95))  # a real, different address (review F6)
        print("P2' pairs per country:", d.group_by("c").len().sort("c").rows(), flush=True)
        out = (d if a.country == "ALL" else d.filter(pl.col("c") == a.country)).select("i", "j", "c", source1_entity_id="s", e2="e2")
        out.write_parquet(a.out)
        print(f"wrote {a.out}: {out.height} {a.country} pairs", flush=True)
        return
    d = d.filter(pl.col("h2") != "")
    nrec = cd.filter(pl.col("h2") != "").group_by("c2", "n2", "h2").len("nrec")
    ns1 = s1.filter(pl.col("h1") != "").group_by(c2="c", n2="n1", h2="h1").len("ns1")
    d = d.join(nrec, on=["c2", "n2", "h2"], how="left").join(ns1, on=["c2", "n2", "h2"], how="left").fill_null(0) \
         .filter((pl.col("nrec") <= 1) & (pl.col("ns1") == 0))
    d = d.with_columns(asim=pl.Series(process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    d = d.filter(pl.col("asim") >= 95)
    print("G-common pairs per country:", d.group_by("c").len().sort("c").rows(), flush=True)
    out = (d if a.country == "ALL" else d.filter(pl.col("c") == a.country)).select("i", "j", "c", source1_entity_id="s", e2="e2")
    out.write_parquet(a.out)
    print(f"wrote {a.out}: {out.height} {a.country} pairs", flush=True)


if __name__ == "__main__":
    main()
