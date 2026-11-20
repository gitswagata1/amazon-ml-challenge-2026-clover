"""Unseen-country check of France rejection rules applied in sequence (e.g. P1 then P2'): train LightGBM on one train
country, score the other, reject the model's accepted pairs in each rule's pair list (experiments/france_rule.py
--split train --country ALL) and report precision + F0.5 change on the unseen country. The arm that matched the France LB
probe is US->India WITHOUT the ce feature.

nice -n 15 python experiments/rule_sim.py --rule P1=work9/p1_train.parquet --rule "P2'=work9/p2s_train.parquet"
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # in the submission zip: src/experiments/ -> src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))  # git repo
from run import PARAMS, decide, f05, truth_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work9"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--rule", action="append", required=True, help="NAME=parquet with i, j (applied in the given order)")
    ap.add_argument("--n-s1", type=int, default=250000)
    ap.add_argument("--t", type=float, default=0.74)
    a = ap.parse_args()
    rules = [(r.split("=", 1)[0], pl.read_parquet(r.split("=", 1)[1], columns=["i", "j"])) for r in a.rule]
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
    truth = truth_pairs(a, str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")).filter(
        pl.col("i").is_in(ids["i"].implode()))
    print(f"{ids.height} S1, {df.height} pairs", flush=True)
    for src, dst, drop in (("US", "India", ["ce"]), ("US", "India", []), ("India", "US", ["ce"]), ("India", "US", [])):
        tr, te = df.filter(pl.col("c") == src), df.filter(pl.col("c") == dst)
        cols = [c for c in df.columns if c not in ("i", "j", "y", "c", *drop)]
        prob = lgb.LGBMClassifier(**PARAMS).fit(tr.select(cols).to_pandas(), tr["y"].to_numpy()).predict_proba(te.select(cols).to_pandas())[:, 1]
        tids = te["i"].unique()
        tt = truth.filter(pl.col("i").is_in(tids.implode()))
        q = decide(te.select("i", "j"), prob, a.t, True)
        prev = f05(q, tt, tids)
        line = f"{src}->{dst} [{'without' if drop else 'with'} ce] F0.5 {prev:.5f}"
        for name, rp in rules:
            rej = q.join(rp, on=["i", "j"], how="semi")
            prec = rej.join(tt, on=["i", "j"], how="semi").height / max(rej.height, 1)
            q = q.join(rp, on=["i", "j"], how="anti")
            s = f05(q, tt, tids)
            line += f" | +{name}: rejects {rej.height}, precision {prec:.3f}, {s:.5f} ({s - prev:+.5f})"
            prev = s
        print(line, flush=True)


if __name__ == "__main__":
    main()
