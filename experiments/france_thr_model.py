"""Fit France's calibration from LB steps and predict the best France threshold (27 Sep).
Raising France's threshold only REMOVES French pairs. For a removed pair in a row with n accepted pairs (others assumed
true): if the pair is false the row's F0.5 rises by g(n) = 1 - F0.5(P=(n-1)/n, R=1) (n=1: row -> empty, a singleton,
+1); if true it falls by l(n) = 1 - F0.5(P=1, R=(n-1)/n) (n=1: -1). With q(p) = sigmoid(a + b*logit p) the chance that a
French pair with model prob p is a true match: dLB = sum over removed pairs of [(1-q) g - q l] / N_S1.
Fit a, b on the measured LB steps (least squares), predict LB for every threshold.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import least_squares

N = 1732544
rd = lambda d: pl.read_csv(f"{d}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)  # noqa: E731
s1 = pl.scan_parquet("work9/test_s1/*.parquet").select(i="idx", s="entity_id", c="country").collect()
cd = pl.scan_parquet("work9/test_cand/*.parquet").select(j="idx", e2="entity_id").collect()
fr = s1.filter(pl.col("c") == "France")


def pairs(d):
    m = rd(d).with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
        .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s", "matched_entity_ids": "e2"})
    return m.join(fr.select("s", "i"), on="s").join(cd, on="e2").select("i", "j")


def f05(P, R):
    return np.where((P + R) > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)


def removed(lo_dir, hi_dir, prob):
    """pairs accepted at the lower threshold but not at the higher one, with g, l, p"""
    lo, hi = pairs(lo_dir), pairs(hi_dir)
    n = lo.group_by("i").len("n")
    r = lo.join(hi, on=["i", "j"], how="anti").join(n, on="i").join(prob, on=["i", "j"], how="left")
    extra = hi.join(lo, on=["i", "j"], how="anti").height
    nn = r["n"].to_numpy().astype(float)
    g = np.where(nn > 1, 1 - f05((nn - 1) / nn, 1.0), 1.0)
    lo_ = np.where(nn > 1, 1 - f05(1.0, (nn - 1) / nn), 1.0)
    p = np.clip(r["prob"].to_numpy(), 1e-6, 1 - 1e-6)
    return dict(g=g, l=lo_, lp=np.log(p / (1 - p)), extra=extra, k=r.height)


def main():
    p5b = pl.read_parquet("work9/test_prob_v9_p5b.parquet").join(fr.select("i"), on="i")
    p5 = pl.read_parquet("work9/test_prob_v9ce3ab_control.parquet").join(fr.select("i"), on="i")
    B = "output_probe_hyb2_p5b"
    obs = [  # (lower file, higher file, prob file, measured dLB)
        (B, B + "_frt0.04", p5b, 0.986128 - 0.98605),
        (B + "_frt0.04", B + "_frt0.08", p5b, 0.986206 - 0.986128),
        (B + "_frt0.08", B + "_frt0.20", p5b, 0.986368 - 0.986206),
        (B + "_frt0.20", B + "_frt0.24", p5b, 0.986191 - 0.986368),
        ("output_probe_hyb_A_fr", "output_probe_hyb_frt0.04", p5, 0.985981 - 0.985926),
    ]
    steps = []
    for lo, hi, pr, d in obs:
        r = removed(lo, hi, pr)
        steps.append((r, d))
        print(f"{lo} -> {hi}: removed {r['k']} pairs (added {r['extra']}), measured dLB {d:+.6f}", flush=True)

    def pred(theta, r):
        q = 1 / (1 + np.exp(-(theta[0] + theta[1] * r["lp"])))
        return ((1 - q) * r["g"] - q * r["l"]).sum() / N

    fit = least_squares(lambda th: np.array([(pred(th, r) - d) * 1e6 for r, d in steps]), x0=[-1.0, 1.0])
    a, b = fit.x
    print(f"fit: q(p) = sigmoid({a:.3f} + {b:.3f} * logit p); residuals (x1e-6 LB): {np.round(fit.fun, 1)}", flush=True)
    for pp in (0.8, 0.9, 0.92, 0.95, 0.98, 0.99):
        print(f"  France P(true | p={pp}) = {1 / (1 + np.exp(-(a + b * np.log(pp / (1 - pp))))):.3f}")
    # predictions for staged candidates relative to the 0.92 file (0.986368)
    base = 0.986368
    for name, d in (("0.90", B + "_frt0.18"), ("0.93", B + "_frt0.21"), ("0.94", B + "_frt0.22"), ("0.96", B + "_frt0.24"),
                    ("0.98", B + "_frt0.26")):
        lo, hi = (d, B + "_frt0.20") if name == "0.90" else (B + "_frt0.20", d)
        r = removed(lo, hi, p5b)
        dl = pred(fit.x, r)
        val = base - dl if name == "0.90" else base + dl
        print(f"predicted LB France {name}: {val:.6f} ({val - base:+.6f} vs 0.92)", flush=True)


if __name__ == "__main__":
    main()
