"""Review handoff2 P2: conflict-aware acceptance. Reject an accepted (one-owner winner) pair when the record's SECOND-best
S1 has prob >= --second and the gap p_top - p_second < delta (near-tie between look-alike S1 -> the winner is right about
half the time, below F0.5's ~0.72 break-even). Checks: (a) in-country v9 OOF per country (must not hurt US/India);
(b) unseen country (train US, score ALL India S1, no ce), on top of P1 + P2'; (c) French test pairs affected.

nice -n 15 python experiments/conflict_sim.py --deltas 0.1,0.2,0.3
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import PARAMS, decide, f05, truth_pairs  # noqa: E402


def conflicts(p, t, delta, second):
    """p: i, j, prob (all scored pairs). Accepted winners whose record has a near-tied runner-up."""
    s = p.sort("prob", descending=True).with_columns(rk=pl.col("prob").rank("ordinal", descending=True).over("j"))
    top = s.filter(pl.col("rk") == 1).select("i", "j", p1="prob")
    sec = s.filter(pl.col("rk") == 2).select("j", p2="prob")
    return top.join(sec, on="j").filter((pl.col("p1") >= t) & (pl.col("p2") >= second) & (pl.col("p1") - pl.col("p2") < delta)).select("i", "j")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work9"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--deltas", default="0.1,0.2,0.3")
    ap.add_argument("--second", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=250000)
    ap.add_argument("--test-prob", type=Path, default=Path("work9/test_prob_v9ce2.parquet"))
    a = ap.parse_args()
    deltas = [float(x) for x in a.deltas.split(",")]
    cty = pl.scan_parquet(str(a.work / "train_s1" / "*.parquet")).select(i="idx", c="country").collect()
    truth = truth_pairs(a, str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet"))
    # (c) French test pairs affected (v9+ce2, t 0.68, France 0.72)
    tc = pl.scan_parquet(str(a.work / "test_s1" / "*.parquet")).select(i="idx", c="country").collect()
    tp = pl.read_parquet(a.test_prob, columns=["i", "j", "prob"])
    for d in deltas:
        n = conflicts(tp, 0.68, d, a.second).join(tc, on="i").group_by("c").len().sort("c").rows()
        print(f"(c) test accepted pairs flagged, delta {d}: {n}", flush=True)
    # (a) in-country: v9 OOF (all 2.1M S1, full competition)
    o = pl.read_parquet(a.work / "oof_v9.parquet").rename({"oof": "prob"})
    ids = o["i"].unique()
    tr_all = truth.filter(pl.col("i").is_in(ids.implode()))
    q = decide(o.select("i", "j"), o["prob"].to_numpy(), 0.74, True)
    for d in deltas:
        rej = conflicts(o.select("i", "j", "prob"), 0.74, d, a.second)
        q2 = q.join(rej, on=["i", "j"], how="anti")
        out = []
        for cc in ("US", "India"):
            ii = cty.filter(pl.col("c") == cc).join(ids.to_frame("i"), on="i")["i"]
            f = lambda qq: f05(qq.filter(pl.col("i").is_in(ii.implode())), tr_all.filter(pl.col("i").is_in(ii.implode())), ii)  # noqa: E731
            out.append(f"{cc} {f(q2) - f(q):+.5f}")
        prec = rej.join(tr_all, on=["i", "j"], how="semi").height / max(rej.height, 1)
        print(f"(a) in-country v9 OOF, delta {d}: rejects {rej.height}, precision {prec:.3f}; " + ", ".join(out), flush=True)
    # (b) unseen: train US (sample), score ALL India S1, no ce; on top of P1 + P2'
    t = json.loads(Path("code/business_entity_resolution/best_params.json").read_text())
    PARAMS.update(num_leaves=t["num_leaves"], min_child_samples=t["min_child_samples"], colsample_bytree=t["feature_fraction"],
                  subsample=t["bagging_fraction"], reg_alpha=t["lambda_l1"], reg_lambda=t["lambda_l2"],
                  scale_pos_weight=t["scale_pos_weight"], n_estimators=500, learning_rate=0.08, n_jobs=16)
    have = pl.scan_parquet(a.work / "train_features.parquet").select("i").unique().collect().join(cty, on="i")
    us = have.filter(pl.col("c") == "US").sample(a.n_train, seed=0)
    ind = have.filter(pl.col("c") == "India")
    def load(sel):
        df = pl.scan_parquet(a.work / "train_features.parquet").filter(pl.col("i").is_in(sel["i"].implode())).collect()
        for pat in ("gpu/train_emb.parquet", "tfw_train.parquet", "rarity_train.parquet"):
            e = pl.scan_parquet(a.work / pat).filter(pl.col("i").is_in(sel["i"].implode())).collect()
            df = df.join(e, on=["i", "j"], how="left", maintain_order="left")
        return df
    tr = load(us)
    cols = [c for c in tr.columns if c not in ("i", "j", "y")]
    m = lgb.LGBMClassifier(**PARAMS).fit(tr.select(cols).to_pandas(), tr["y"].to_numpy())
    del tr
    te = load(ind)
    prob = m.predict_proba(te.select(cols).to_pandas())[:, 1]
    pp = te.select("i", "j").with_columns(prob=pl.Series(prob))
    del te
    tids = ind["i"]
    tt = truth.filter(pl.col("i").is_in(tids.implode()))
    q = decide(pp.select("i", "j"), prob, 0.74, True)
    base = f05(q, tt, tids)
    for f in ("p1_train.parquet", "p2s_train.parquet"):
        q = q.join(pl.read_parquet(a.work / f, columns=["i", "j"]), on=["i", "j"], how="anti")
    after = f05(q, tt, tids)
    print(f"(b) US->ALL India (no ce): F0.5 {base:.5f}; after P1+P2' {after:.5f} ({after - base:+.5f})", flush=True)
    for d in deltas:
        rej = conflicts(pp, 0.74, d, a.second).join(q, on=["i", "j"], how="semi")
        prec = rej.join(tt, on=["i", "j"], how="semi").height / max(rej.height, 1)
        s = f05(q.join(rej, on=["i", "j"], how="anti"), tt, tids)
        print(f"(b) delta {d}: rejects {rej.height}, precision {prec:.3f}, {s:.5f} ({s - after:+.5f})", flush=True)


if __name__ == "__main__":
    main()
