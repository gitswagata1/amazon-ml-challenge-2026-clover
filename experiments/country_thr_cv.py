"""Per-country decision search on a stage-2 OOF (v12 L1): for each train country, the threshold t and the conflict-rule
delta that maximise that country's macro F0.5 (all sampled S1 of the country, tune_meta.parquet), with exactly the
test-side logic of country_threshold.py (one-owner = highest prob per record; conflict rule = drop the winner when the
record's 2nd-best S1 has prob >= 0.5 and the gap is < delta). Records belong to one country, so countries are separable.

python experiments/country_thr_cv.py --oof work10/oof_v10ce2.parquet --work work10 --t 0.68 --delta 0.2
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl


def f05(acc, meta):
    """Macro F0.5 over meta's S1 (i, ntrue); acc: accepted pairs i, y."""
    g = acc.group_by("i").agg(npred=pl.len(), tp=pl.col("y").sum().cast(pl.Float64))
    per = meta.join(g, on="i", how="left").fill_null(0)
    P, R = pl.col("tp") / pl.col("npred"), pl.col("tp") / pl.col("ntrue")
    return per.select(pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
                      .otherwise(1.25 * P * R / (0.25 * P + R)).mean()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, required=True, help="i, j, y, oof (stage-2 OOF from ce2_band.py --save-oof)")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--meta", type=Path, help="default <work>/tune_meta.parquet")
    ap.add_argument("--t", type=float, required=True, help="current common threshold")
    ap.add_argument("--delta", type=float, required=True, help="current conflict delta")
    a = ap.parse_args()
    o = pl.read_parquet(a.oof, columns=["i", "j", "y", "oof"])
    meta = pl.read_parquet(a.meta or a.work / "tune_meta.parquet", columns=["i", "ntrue"])
    c = pl.scan_parquet(str(a.work / "train_s1" / "*.parquet")).select(i="idx", c="country").collect()
    second = (o.sort("oof", descending=True).with_columns(rk=pl.int_range(pl.len()).over("j"))
              .filter(pl.col("rk") == 1).select("j", p2="oof"))
    top = (o.filter(pl.col("oof") == pl.col("oof").max().over("j")).join(second, on="j", how="left")
           .with_columns(pl.col("p2").fill_null(0.0)).join(c, on="i"))
    meta = meta.join(c, on="i")
    grid_t = np.round(np.arange(0.40, 0.86, 0.02), 2)
    grid_d = [0.0, 0.15, 0.2, 0.25, 0.3, 0.35]
    tot_cur = tot_best = 0.0
    for country in sorted(meta["c"].unique().to_list()):
        tc, mc = top.filter(pl.col("c") == country), meta.filter(pl.col("c") == country)
        res = {}
        for d in grid_d:
            near = (pl.col("p2") >= 0.5) & (pl.col("oof") - pl.col("p2") < d)
            base = tc.filter(~near)
            for t in grid_t:
                res[(float(t), d)] = f05(base.filter(pl.col("oof") >= t), mc)
        cur = res.get((a.t, a.delta)) or f05(tc.filter((pl.col("oof") >= a.t) & ~((pl.col("p2") >= 0.5) & (pl.col("oof") - pl.col("p2") < a.delta))), mc)
        (bt, bd), bv = max(res.items(), key=lambda kv: kv[1])
        w = mc.height / meta.height
        tot_cur += w * cur
        tot_best += w * bv
        top5 = sorted(res.items(), key=lambda kv: -kv[1])[:5]
        print(f"{country}: {mc.height} S1 | current t={a.t} delta={a.delta}: {cur:.5f} | best t={bt} delta={bd}: {bv:.5f} "
              f"({bv - cur:+.5f}) | top5 {[(k, round(v, 5)) for k, v in top5]}", flush=True)
    print(f"ALL: current {tot_cur:.5f} -> per-country best {tot_best:.5f} ({tot_best - tot_cur:+.5f})", flush=True)


if __name__ == "__main__":
    main()
