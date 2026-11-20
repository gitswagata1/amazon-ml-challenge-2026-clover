"""N2 error breakdown + N1 second-stage (list-context) test on a matcher's out-of-fold predictions.

N2: where macro F0.5 is lost: true matches never in the candidates, false positives, false negatives in candidates.
N1: 2nd-stage LightGBM on [logit(oof)] (control) vs [logit(oof) + list-context features], 5-fold GroupKFold by S1.

python experiments/n1_n2.py --oof work/oof.parquet --n-train 600000
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
import run  # noqa: E402

THR = np.arange(0.5, 0.9, 0.02)


def context(df):
    """List-context features from the first-stage probability p (per S1 i and per candidate j)."""
    p = pl.col("p")
    return df.with_columns(
        lo=(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6))).log(),
        rank_i=p.rank("ordinal", descending=True).over("i"),
        gap_i=p.max().over("i") - p,
        n50_i=(p >= 0.5).sum().over("i"), n20_i=(p >= 0.2).sum().over("i"), n80_i=(p >= 0.8).sum().over("i"),
        sum_i=p.sum().over("i"),
        rank_j=p.rank("ordinal", descending=True).over("j"),
        gap_j=p.max().over("j") - p,
        n50_j=(p >= 0.5).sum().over("j"), cnt_j=pl.len().over("j"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="work/oof.parquet")
    ap.add_argument("--n-train", type=int, default=600_000)
    a = ap.parse_args()
    df = pl.read_parquet(a.oof).rename({"oof": "p"}).sort("i")
    s1p = "work/train_s1/*.parquet"
    ids = run.train_sample(SimpleNamespace(n_train=a.n_train, exclude_s1=None), s1p)
    truth = pl.read_parquet("work/train_truth.parquet").filter(pl.col("i").is_in(ids.implode()))
    ctry = pl.scan_parquet(s1p).select(pl.col("idx").alias("i"), "ctry").collect()
    f = lambda pred, tr=truth, ii=ids: run.f05(pred, tr, ii)
    best = lambda prob: max((f(run.decide(df, prob, t, True)), round(float(t), 2)) for t in THR)

    # ---- N2 ----
    s, t = best(df["p"].to_numpy())
    pred = run.decide(df, df["p"].to_numpy(), t, True)
    inc = df.filter(pl.col("y") == 1).select("i", "j")
    print(f"N2  actual {s:.5f} @ {t}")
    print(f"    perfect model on these candidates (ceiling): {f(inc):.5f}   -> lost to never-found: {1 - f(inc):.5f}")
    print(f"    remove all false positives: {f(pred.join(truth, on=['i', 'j'])):.5f}")
    print(f"    add all missed true pairs in candidates: {f(pl.concat([pred, inc]).unique()):.5f}")
    for c in ("us", "india"):
        ic = ctry.filter(pl.col("ctry") == c)["i"]
        ic = ids.filter(ids.is_in(ic.implode()))
        tc = truth.filter(pl.col("i").is_in(ic.implode()))
        print(f"    {c}: actual {run.f05(pred.filter(pl.col('i').is_in(ic.implode())), tc, ic):.5f}, "
              f"ceiling {run.f05(inc.filter(pl.col('i').is_in(ic.implode())), tc, ic):.5f}, S1 {len(ic)}")
    per = (ids.to_frame("i").join(truth.group_by("i").len("ntrue"), on="i", how="left").fill_null(0)
           .join(pred.group_by("i").len("npred"), on="i", how="left").fill_null(0))
    print(f"    singletons (no true match): {(per['ntrue'] == 0).sum()} S1, wrongly given a match: "
          f"{((per['ntrue'] == 0) & (per['npred'] > 0)).sum()}; with matches but predicted none: "
          f"{((per['ntrue'] > 0) & (per['npred'] == 0)).sum()}", flush=True)

    # ---- N1 ----
    d = context(df)
    y, g = d["y"].to_numpy(), d["i"].to_numpy()
    cols = [c for c in d.columns if c not in ("i", "j", "y", "p")]
    for name, X in (("control [p]", d.select("lo").to_numpy()), ("N1 [p + context]", d.select(cols).to_numpy())):
        p2 = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(X, y, g):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1)
            p2[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        print(f"N1  {name}: {best(p2)}", flush=True)


if __name__ == "__main__":
    main()
