"""Unseen-country corrector (v11 idea): learn, from an unseen-country proxy, how a model's accepted/borderline pairs go
wrong on a country it never saw (generalising the LB-proven France rules P1/P2' from hand rules to a small model).
Proxy = LightGBM without the ce feature trained on one train country, scoring the other (experiments/proxy_cache.py);
the US->India proxy matched the France LB probe. Train the corrector on one direction, evaluate on the OTHER direction
(held-out country): one-owner decisions with corrected probs vs raw, and vs raw + P1 + P2'.
Features are label-free: logit(p), 2nd-best S1 prob of the record, #S1 claiming it (p >= 0.5), name x address
relation (proxy_mine.categorise), rarity counts, candidate source.

nice -n 15 python experiments/corrector_sim.py
"""
import sys
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # in the submission zip: src/experiments/ -> src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))  # git repo
from proxy_mine import categorise, W  # noqa: E402
from run import decide, f05, truth_pairs  # noqa: E402

P_MIN, T = 0.05, 0.74
CATS = ["name", "addr", "src"]


def features(p, split):
    """p: i, j, prob (+ y) for ALL pairs of the scored S1 sample. Returns the p >= P_MIN rows with features."""
    ctx = (p.sort("prob", descending=True).with_columns(rk=pl.int_range(pl.len()).over("j"))
           .group_by("j").agg(p2=pl.col("prob").filter(pl.col("rk") == 1).first(),
                              claims=(pl.col("prob") >= 0.5).sum()))
    q = p.filter(pl.col("prob") >= P_MIN).join(ctx, on="j", how="left").with_columns(pl.col("p2").fill_null(0.0))
    cat = categorise(q.select("i", "j"), split)
    src = pl.scan_parquet(str(W / f"{split}_cand" / "*.parquet")).select(j="idx", src=pl.col("entity_id").str.slice(0, 2)).collect()
    rar = pl.read_parquet(W / f"rarity_{split}.parquet", columns=["i", "j", "name_df_s1", "name_df_cand", "cand_addr_empty"])
    q = q.join(cat.select("i", "j", "name", "addr"), on=["i", "j"], how="left").join(src, on="j", how="left") \
         .join(rar, on=["i", "j"], how="left")
    pc = q["prob"].clip(1e-6, 1 - 1e-6).to_numpy()
    return q.with_columns(lp=pl.Series(np.log(pc / (1 - pc)), dtype=pl.Float32), gap=pl.col("prob") - pl.col("p2"))


FEATS = ["lp", "p2", "gap", "claims", "name_df_s1", "name_df_cand", "cand_addr_empty"] + CATS


def xy(q):
    X = q.select(FEATS).to_pandas()
    for c in CATS:
        X[c] = X[c].astype("category")
    return X


def main():
    a = types.SimpleNamespace(data=Path("dataset"))
    truth_all = truth_pairs(a, str(W / "train_s1" / "*.parquet"), str(W / "train_cand" / "*.parquet"))
    rules = pl.concat([pl.read_parquet(W / f, columns=["i", "j"]) for f in ("p1_train.parquet", "p2s_train.parquet")]).unique()
    data = {}
    for name in ("us2india", "india2us"):
        p = pl.read_parquet(W / f"proxy_{name}.parquet")
        data[name] = (p, features(p, "train"))
        print(f"{name}: {p.height} pairs, {data[name][1].height} with p >= {P_MIN}", flush=True)
    for tr_name, te_name in (("us2india", "india2us"), ("india2us", "us2india")):
        _, qtr = data[tr_name]
        p, qte = data[te_name]
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=200,
                               verbose=-1, n_jobs=8).fit(xy(qtr), qtr["y"].to_numpy())
        corr = qte.select("i", "j").with_columns(pc=pl.Series(m.predict_proba(xy(qte))[:, 1], dtype=pl.Float32))
        pp = p.join(corr, on=["i", "j"], how="left").with_columns(pc=pl.coalesce("pc", "prob"))
        ids = p["i"].unique()
        tt = truth_all.filter(pl.col("i").is_in(ids.implode()))
        raw = decide(p.select("i", "j"), p["prob"].to_numpy(), T, True)
        res = {"raw": f05(raw, tt, ids), "raw+P1+P2'": f05(raw.join(rules, on=["i", "j"], how="anti"), tt, ids)}
        for t in (0.6, 0.7, 0.74, 0.8, 0.85):
            res[f"corrector t={t}"] = f05(decide(pp.select("i", "j"), pp["pc"].to_numpy(), t, True), tt, ids)
        base = res["raw"]
        print(f"train corrector on {tr_name}, evaluate on held-out {te_name}: " +
              " | ".join(f"{k} {v:.5f} ({v - base:+.5f})" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
