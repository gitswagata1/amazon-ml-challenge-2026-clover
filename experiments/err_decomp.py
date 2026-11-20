"""Where does a model lose macro F0.5? From run.py train's OOF predictions (work/oof.parquet) at the CV threshold:
ceiling (perfect decisions on the candidates), FP-free, FN-free, per country, and the loss split by S1 type
(singleton with a false match / partial / fully missed). Also estimates France's public-LB score from the
per-country CV and the test's country mix.

python experiments/err_decomp.py --work work --t 0.72 --lb 0.974
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import decide, f05, truth_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--t", type=float, default=0.72)
    ap.add_argument("--lb", type=float, default=0.974, help="public LB of this model, to back out France")
    a = ap.parse_args()
    s1p, cp = str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")
    o = pl.read_parquet(a.work / "oof.parquet")
    ids = o["i"].unique()
    truth = truth_pairs(a, s1p, cp).filter(pl.col("i").is_in(ids.implode()))
    pred = decide(o, o["oof"].to_numpy(), a.t, True)
    cand_truth = truth.join(o.select("i", "j"), on=["i", "j"], how="semi")
    ctry = pl.scan_parquet(s1p).select(pl.col("idx").alias("i"), "ctry").collect()

    variants = {"actual": pred, "ceiling (perfect on candidates)": cand_truth,
                "FP-free": pred.join(truth, on=["i", "j"], how="semi"),
                "FN-free (on candidates)": pl.concat([pred, cand_truth]).unique()}
    groups = {"all": ids} | {c: ids.to_frame().join(ctry, on="i").filter(pl.col("ctry") == c)["i"]
                             for c in ctry["ctry"].unique().sort()}
    res = {}
    print(f"{'':32s}" + "".join(f"{g:>10s}" for g in groups))
    for name, pr in variants.items():
        row = [f05(pr, truth, g) for g in groups.values()]
        res[name] = dict(zip(groups, row))
        print(f"{name:32s}" + "".join(f"{x:10.5f}" for x in row))
    print("S1 share:" + "".join(f"  {g}={len(v) / len(ids):.3f}" for g, v in groups.items() if g != "all"))

    # loss by S1 type (share of the total 1 - F0.5)
    per = (ids.to_frame("i").join(pred.group_by("i").len("npred"), on="i", how="left")
           .join(truth.group_by("i").len("ntrue"), on="i", how="left")
           .join(pred.join(truth, on=["i", "j"]).group_by("i").len("tp"), on="i", how="left").fill_null(0)
           .join(cand_truth.group_by("i").len("ncand_true"), on="i", how="left").fill_null(0).join(ctry, on="i"))
    P, R = pl.col("tp") / pl.col("npred"), pl.col("tp") / pl.col("ntrue")
    per = per.with_columns(score=pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0)
                           .when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * P * R / (0.25 * P + R)))
    kind = (pl.when(pl.col("ntrue") == 0).then(pl.lit("singleton, predicted a match"))
            .when(pl.col("ncand_true") == 0).then(pl.lit("matches exist, none in candidates"))
            .when(pl.col("tp") == 0).then(pl.lit("matches in candidates, none predicted"))
            .otherwise(pl.lit("partly right (extra/missing ids)")))
    loss = (per.with_columns(kind=kind, loss=1 - pl.col("score")).filter(pl.col("loss") > 0)
            .group_by("ctry", "kind").agg(n=pl.len(), loss=pl.col("loss").sum() / len(ids)).sort("loss", descending=True))
    print("\nF0.5 lost per S1 type (points of the overall score):")
    print(loss.with_columns(pl.col("loss").round(5)))
    print(f"singletons: {(per['ntrue'] == 0).mean():.3f} of S1")

    # France backed out of the public LB, assuming US/India score on the LB what they score in CV
    test_mix = {"us": 663106, "india": 809986, "france": 259452}
    known = sum(res["actual"][c] * test_mix[c] for c in ("us", "india") if c in res["actual"])
    fr = (a.lb * sum(test_mix.values()) - known) / test_mix["france"]
    print(f"\nimplied France score on the public LB (LB {a.lb}): {fr:.4f}  "
          f"(US/India at test mix: {known / (test_mix['us'] + test_mix['india']):.4f})")


if __name__ == "__main__":
    main()
