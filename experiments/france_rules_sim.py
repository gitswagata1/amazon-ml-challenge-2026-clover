"""Screen further 'generic word' rejection rules on the unseen-country proxy that matched the France LB probe
(train US only, score India WITHOUT the ce feature: predicted +0.006 on the country for rule P1; France LB gave +0.0075).
For each pattern: precision of the model's accepted pairs and the effect of rejecting them on India macro F0.5, applied
on top of P1 (the adopted rule). Also counts the French test pairs each rule would remove (--test-prob).

nice -n 15 python experiments/france_rules_sim.py --test-prob work9/test_prob_v9ce2.parquet --t-test 0.68
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import PARAMS, decide, f05, truth_pairs  # noqa: E402

RULES = ("P1 same-addr 1 common swap, single rec", "P2 1 common swap, addr differs", "P3 same-addr 1 common swap, twin/S1-sib",
         "P4 +1 common word", "P5 -1 common word", "P6 >=2 common-word changes")


def flags(pairs, work, split):
    """pairs: i, j (+ anything). Adds boolean columns, one per rule (label-free, per split+country word stats)."""
    s1 = pl.scan_parquet(str(work / f"{split}_s1" / "*.parquet")).select(i="idx", c="country", n1="core", a1="addr", h1="hn").collect()
    cd = pl.scan_parquet(str(work / f"{split}_cand" / "*.parquet")).select(j="idx", n2="core", a2="addr", h2="hn").collect()
    wdf = (s1.with_columns(w=pl.col("n1").str.split(" ").list.unique()).explode("w").group_by("c", "w").len("df")
           .filter(pl.col("df") >= 20).select("c", "w"))
    common = {c: set(g["w"]) for (c,), g in wdf.group_by("c")}
    twin = cd.join(s1.select("i", "c").unique("i").head(0), on="i", how="left") if False else None  # noqa
    ctry = pl.scan_parquet(str(work / f"{split}_cand" / "*.parquet")).select(j="idx", c2="country").collect()
    cdt = cd.join(ctry, on="j").filter(pl.col("h2") != "")
    nrec = cdt.group_by("c2", "n2", "h2").len("nrec")
    ns1 = s1.filter(pl.col("h1") != "").group_by(c2="c", n2="n1", h2="h1").len("ns1")
    d = pairs.join(s1, on="i").join(cd, on="j").join(nrec, left_on=["c", "n2", "h2"], right_on=["c2", "n2", "h2"], how="left") \
        .join(ns1, left_on=["c", "n2", "h2"], right_on=["c2", "n2", "h2"], how="left").fill_null(0)
    asim = process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    out = []
    for c, n1, n2, h2, s, nr, nsib in zip(d["c"], d["n1"], d["n2"], d["h2"], asim, d["nrec"], d["ns1"]):
        t1, t2 = set((n1 or "").split()), set((n2 or "").split())
        ra, rb, k, cm = t1 - t2, t2 - t1, len(t1 & t2), common.get(c, set())
        same = s >= 95 and h2 != ""
        swap = len(ra) == 1 and len(rb) == 1 and k >= 1 and all(w in cm for w in ra | rb) \
            and fuzz.ratio(next(iter(ra)), next(iter(rb))) < 70
        single = nr <= 1 and nsib == 0
        out.append((swap and same and single, swap and not same, swap and same and not single,
                    not ra and len(rb) == 1 and k >= 1 and all(w in cm for w in rb),
                    not rb and len(ra) == 1 and k >= 1 and all(w in cm for w in ra),
                    k >= 1 and len(ra) + len(rb) >= 3 and ra and rb and all(w in cm for w in ra | rb)))
    arr = np.array(out, dtype=bool).reshape(-1, len(RULES))
    return d.select("i", "j", "c").with_columns([pl.Series(r, arr[:, n]) for n, r in enumerate(RULES)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work9"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--n-s1", type=int, default=250000)
    ap.add_argument("--t", type=float, default=0.74)
    ap.add_argument("--test-prob", type=Path)
    ap.add_argument("--t-test", type=float, default=0.68)
    a = ap.parse_args()
    if a.test_prob:  # French volumes on the test side (accepted, one-owner)
        tp = pl.read_parquet(a.test_prob, columns=["i", "j", "prob"])
        tp = tp.filter(pl.col("prob") == pl.col("prob").max().over("j")).filter(pl.col("prob") >= a.t_test)
        f = flags(tp.select("i", "j"), a.work, "test")
        print("test accepted pairs per rule:", f.group_by("c").agg([pl.col(r).sum() for r in RULES]).sort("c").to_dicts(), flush=True)
        f.filter(pl.col("c") == "France").write_parquet(a.work / "france_rule_flags.parquet")
    t = json.loads(Path("code/business_entity_resolution/best_params.json").read_text())
    PARAMS.update(num_leaves=t["num_leaves"], min_child_samples=t["min_child_samples"], colsample_bytree=t["feature_fraction"],
                  subsample=t["bagging_fraction"], reg_alpha=t["lambda_l1"], reg_lambda=t["lambda_l2"],
                  scale_pos_weight=t["scale_pos_weight"], n_estimators=500, learning_rate=0.08, n_jobs=16)
    cty = pl.scan_parquet(str(a.work / "train_s1" / "*.parquet")).select(i="idx", c="country").collect()
    have = pl.scan_parquet(a.work / "train_features.parquet").select("i").unique().collect().join(cty, on="i")
    ids = pl.concat([have.filter(pl.col("c") == c).sample(a.n_s1, seed=0) for c in ("US", "India")])
    df = pl.scan_parquet(a.work / "train_features.parquet").filter(pl.col("i").is_in(ids["i"].implode())).collect()
    for pat in ("gpu/train_emb.parquet", "tfw_train.parquet", "rarity_train.parquet"):  # no ce: the proxy that matched France
        e = pl.scan_parquet(a.work / pat).filter(pl.col("i").is_in(ids["i"].implode())).collect()
        df = df.join(e, on=["i", "j"], how="left", maintain_order="left")
    df = df.join(ids, on="i")
    truth = truth_pairs(a, str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")).filter(
        pl.col("i").is_in(ids["i"].implode()))
    tr, te = df.filter(pl.col("c") == "US"), df.filter(pl.col("c") == "India")
    cols = [c for c in df.columns if c not in ("i", "j", "y", "c")]
    prob = lgb.LGBMClassifier(**PARAMS).fit(tr.select(cols).to_pandas(), tr["y"].to_numpy()).predict_proba(te.select(cols).to_pandas())[:, 1]
    tids = te["i"].unique()
    tt = truth.filter(pl.col("i").is_in(tids.implode()))
    q = decide(te.select("i", "j"), prob, a.t, True)
    fq = flags(q, a.work, "train")
    base = f05(q, tt, tids)
    p1 = fq.filter(pl.col(RULES[0])).select("i", "j")
    after1 = f05(q.join(p1, on=["i", "j"], how="anti"), tt, tids)
    print(f"US->India (no ce) F0.5 {base:.5f}; after P1 {after1:.5f} ({after1 - base:+.5f})", flush=True)
    for r in RULES[1:]:
        rej = fq.filter(pl.col(r)).select("i", "j")
        prec = rej.join(tt, on=["i", "j"], how="semi").height / max(rej.height, 1)
        s = f05(q.join(p1, on=["i", "j"], how="anti").join(rej, on=["i", "j"], how="anti"), tt, tids)
        print(f"  {r:42s}: accepted {rej.height:6d}, precision {prec:.3f}, reject on top of P1 {s:.5f} ({s - after1:+.5f})", flush=True)


if __name__ == "__main__":
    main()
