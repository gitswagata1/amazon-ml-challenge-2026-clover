"""Simulate France probe 1 on an unseen country (leave-one-country-out): train LightGBM on one train country only,
score the other as if unseen, then measure the effect of REJECTING the accepted group-G pairs (same address, one common
word swapped, no twin record, no S1 with that name) on the unseen country's macro F0.5. Arms: with / without the `ce`
feature (the CE saw both countries, so the two arms bracket France, whose CE never saw it).

nice -n 15 python experiments/probe1_sim.py --g <twin_train.parquet>
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # in the submission zip: src/experiments/ -> src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))  # git repo
from run import PARAMS, decide, f05, truth_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work9"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--g", type=Path, required=True)
    ap.add_argument("--n-s1", type=int, default=250000)
    ap.add_argument("--t", type=float, default=0.74)
    a = ap.parse_args()
    t = json.loads(Path("code/business_entity_resolution/best_params.json").read_text())
    PARAMS.update(num_leaves=t["num_leaves"], min_child_samples=t["min_child_samples"], colsample_bytree=t["feature_fraction"],
                  subsample=t["bagging_fraction"], reg_alpha=t["lambda_l1"], reg_lambda=t["lambda_l2"],
                  scale_pos_weight=t["scale_pos_weight"], n_estimators=500, learning_rate=0.08, n_jobs=16)
    cty = pl.scan_parquet(str(a.work / "train_s1" / "*.parquet")).select(i="idx", c="country").collect()
    have = pl.scan_parquet(a.work / "train_features.parquet").select("i").unique().collect().join(cty, on="i")
    ids = pl.concat([have.filter(pl.col("c") == c).sample(a.n_s1, seed=0) for c in ("US", "India")])
    df = pl.scan_parquet(a.work / "train_features.parquet").filter(pl.col("i").is_in(ids["i"].implode())).collect()
    for pat in ("gpu/train_emb.parquet", "gpu/train_ce.parquet", "tfw_train.parquet", "rarity_train.parquet"):
        e = pl.scan_parquet(a.work / pat).filter(pl.col("i").is_in(ids["i"].implode())).collect()
        df = df.join(e, on=["i", "j"], how="left", maintain_order="left")
    df = df.join(ids, on="i")
    # group G, common<->common words (df >= 20 among the country's S1 names)
    s1 = pl.scan_parquet(str(a.work / "train_s1" / "*.parquet")).select(c="country", n1="core").collect()
    wdf = s1.with_columns(w=pl.col("n1").str.split(" ").list.unique()).explode("w").group_by("c", "w").len("df")
    g = (pl.read_parquet(a.g).filter((pl.col("kind") == "single record") & ~pl.col("noise"))
         .join(wdf.rename({"w": "add", "df": "df_add"}), on=["c", "add"], how="left")
         .join(wdf.rename({"w": "rem", "df": "df_rem"}), on=["c", "rem"], how="left").fill_null(0)
         .with_columns(gen=(pl.col("df_add") >= 20) & (pl.col("df_rem") >= 20)).select("i", "j", "gen"))
    truth = truth_pairs(a, str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")).filter(
        pl.col("i").is_in(ids["i"].implode()))
    print(f"{ids.height} S1, {df.height} pairs", flush=True)
    for src, dst in (("US", "India"), ("India", "US")):
        tr, te = df.filter(pl.col("c") == src), df.filter(pl.col("c") == dst)
        tids = te["i"].unique()
        tt = truth.filter(pl.col("i").is_in(tids.implode()))
        gg = te.select("i", "j").join(g, on=["i", "j"], how="left")
        for arm, drop in (("with ce", []), ("without ce", ["ce"])):
            cols = [c for c in df.columns if c not in ("i", "j", "y", "c", *drop)]
            m = lgb.LGBMClassifier(**PARAMS).fit(tr.select(cols).to_pandas(), tr["y"].to_numpy())
            prob = m.predict_proba(te.select(cols).to_pandas())[:, 1]
            q = decide(te.select("i", "j"), prob, a.t, True)
            base = f05(q, tt, tids)
            qg = q.join(gg.filter(pl.col("gen").is_not_null()), on=["i", "j"], how="left")
            for name, flt in (("G common<->common", pl.col("gen") == True), ("G all", pl.col("gen").is_not_null())):  # noqa: E712
                acc = qg.filter(flt).select("i", "j")
                prec = acc.join(tt, on=["i", "j"], how="semi").height / max(acc.height, 1)
                rej = f05(q.join(acc, on=["i", "j"], how="anti"), tt, tids)
                print(f"{src}->{dst} [{arm}] F0.5 {base:.5f} | {name}: accepted {acc.height}, precision {prec:.3f}; "
                      f"probe-1 sim (reject them): {rej:.5f} ({rej - base:+.5f})", flush=True)


if __name__ == "__main__":
    main()
