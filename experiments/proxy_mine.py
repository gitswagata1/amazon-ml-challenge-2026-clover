"""v11 France rule mining: categorise the ACCEPTED pairs of the unseen-country proxy (after the adopted P1 + P2' rules) by
label-free name/address relation, and compare each category's precision with the same category in-country (v9 OOF).
Categories where the unseen country is much worse (precision < ~0.72 = F0.5 break-even) are rule candidates.
Also: the French test volume of each category (v9+ce2 accepted, France), to avoid P4-style composition surprises.

python experiments/proxy_mine.py --proxy work9/proxy_us2india.parquet --country India
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

W = Path("work9")


def categorise(p, split):
    s1all = pl.scan_parquet(str(W / f"{split}_s1" / "*.parquet")).select(i="idx", c="country", n1="core")
    s1 = (pl.scan_parquet(str(W / f"{split}_s1" / "*.parquet")).select(i="idx", c="country", n1="core", a1="addr", h1="hn")
          .filter(pl.col("i").is_in(p["i"].unique().implode())).collect())
    cd = (pl.scan_parquet(str(W / f"{split}_cand" / "*.parquet")).select(j="idx", n2="core", a2="addr", h2="hn")
          .filter(pl.col("j").is_in(p["j"].unique().implode())).collect())  # only the rows needed (RAM)
    wdf = (s1all.collect().with_columns(w=pl.col("n1").str.split(" ").list.unique()).explode("w").group_by("c", "w").len("df")
           .filter(pl.col("df") >= 20).select("c", "w"))
    d = p.join(s1, on="i").join(cd, on="j")
    d = d.with_columns(t1=pl.col("n1").str.split(" ").list.unique(), t2=pl.col("n2").str.split(" ").list.unique())
    d = d.with_columns(ra=pl.col("t1").list.set_difference("t2"), rb=pl.col("t2").list.set_difference("t1"),
                       k=pl.col("t1").list.set_intersection("t2").list.len())
    # "all differing words are common" via exploded joins (attaching the whole word list per row blew RAM to 40 GB)
    d = d.with_row_index("r")
    cm = wdf.with_columns(cm=pl.lit(True))
    for col in ("ra", "rb"):
        e = (d.select("r", "c", w=pl.col(col)).explode("w").drop_nulls("w").join(cm, on=["c", "w"], how="left")
             .group_by("r").agg(pl.col("cm").is_not_null().all().alias(f"{col}_c")))
        d = d.join(e, on="r", how="left").with_columns(pl.col(f"{col}_c").fill_null(True))
    nra, nrb = pl.col("ra").list.len(), pl.col("rb").list.len()
    name = (pl.when((nra == 0) & (nrb == 0)).then(pl.lit("same"))
            .when(pl.col("k") == 0).then(pl.lit("no shared word"))
            .when((nra == 1) & (nrb == 1)).then(pl.when(pl.col("ra_c") & pl.col("rb_c")).then(pl.lit("swap1 common")).otherwise(pl.lit("swap1 rare")))
            .when(nra == 0).then(pl.when(pl.col("rb_c")).then(pl.lit("add common")).otherwise(pl.lit("add rare")))
            .when(nrb == 0).then(pl.when(pl.col("ra_c")).then(pl.lit("drop common")).otherwise(pl.lit("drop rare")))
            .otherwise(pl.when(pl.col("ra_c") & pl.col("rb_c")).then(pl.lit("multi common")).otherwise(pl.lit("multi rare"))))
    d = d.with_columns(name=name)
    asim = process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    num = lambda h: h.str.extract(r"^(\d+)", 1)  # noqa: E731
    d = d.with_columns(asim=pl.Series(asim), n1h=num(pl.col("h1")), n2h=num(pl.col("h2")))
    addr = (pl.when(pl.col("a2") == "").then(pl.lit("cand empty"))
            .when(pl.col("asim") >= 95).then(pl.when(pl.col("n1h") == pl.col("n2h")).then(pl.lit("same")).otherwise(pl.lit("same street, num differs")))
            .when(pl.col("asim") >= 80).then(pl.lit("similar"))
            .otherwise(pl.lit("different")))
    return d.with_columns(addr=addr).select("i", "j", "c", "name", "addr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", type=Path, required=True)
    ap.add_argument("--country", required=True, help="the proxy's unseen country")
    ap.add_argument("--t", type=float, default=0.74)
    a = ap.parse_args()
    rules = pl.concat([pl.read_parquet(W / f, columns=["i", "j"]) for f in ("p1_train.parquet", "p2s_train.parquet")]).unique()

    def accepted(p, t):
        q = p.filter(pl.col("prob") == pl.col("prob").max().over("j")).filter(pl.col("prob") >= t)
        return q.join(rules, on=["i", "j"], how="anti")
    px = accepted(pl.read_parquet(a.proxy), a.t)
    cty = pl.scan_parquet(str(W / "train_s1" / "*.parquet")).select(i="idx", c="country").collect()
    ic = pl.read_parquet(W / "oof_v9.parquet").rename({"oof": "prob"}).join(cty, on="i").filter(pl.col("c") == a.country).drop("c")
    ic = accepted(ic, 0.74)
    tp = pl.read_parquet(W / "test_prob_v9ce2.parquet")
    tc = pl.scan_parquet(str(W / "test_s1" / "*.parquet")).select(i="idx", c="country").collect()
    fr = tp.join(tc, on="i").filter(pl.col("c") == "France").drop("c")
    fr = fr.filter(pl.col("prob") == pl.col("prob").max().over("j")).filter(pl.col("prob") >= 0.72)
    frr = pl.concat([pl.read_parquet(W / f, columns=["source1_entity_id", "e2"]) for f in ("g_test_common.parquet", "g2_test_diffaddr_sym.parquet")])
    ids = lambda g: pl.scan_parquet(g).select("idx", "entity_id").collect()  # noqa: E731
    frr = (frr.join(ids(str(W / "test_s1" / "*.parquet")).rename({"idx": "i", "entity_id": "source1_entity_id"}), on="source1_entity_id")
           .join(ids(str(W / "test_cand" / "*.parquet")).rename({"idx": "j", "entity_id": "e2"}), on="e2").select("i", "j"))
    fr = fr.join(frr, on=["i", "j"], how="anti")
    cp = categorise(px.select("i", "j", "y"), "train").join(px.select("i", "j", "y"), on=["i", "j"])
    ci = categorise(ic.select("i", "j", "y"), "train").join(ic.select("i", "j", "y"), on=["i", "j"])
    cf = categorise(fr.select("i", "j"), "test")
    a1 = cp.group_by("name", "addr").agg(n_proxy=pl.len(), prec_proxy=pl.col("y").mean())
    a2 = ci.group_by("name", "addr").agg(n_incountry=pl.len(), prec_incountry=pl.col("y").mean())
    a3 = cf.group_by("name", "addr").agg(n_france=pl.len())
    t = a1.join(a2, on=["name", "addr"], how="full", coalesce=True).join(a3, on=["name", "addr"], how="full", coalesce=True).fill_null(0)
    s1n = cp["i"].n_unique()
    t = t.with_columns(fp_proxy_per1k=(pl.col("n_proxy") * (1 - pl.col("prec_proxy")) / s1n * 1000).round(2),
                       prec_proxy=pl.col("prec_proxy").round(3), prec_incountry=pl.col("prec_incountry").round(3),
                       fr_per1k=(pl.col("n_france") / 259452 * 1000).round(1)).sort("fp_proxy_per1k", descending=True)
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(t)


if __name__ == "__main__":
    main()
