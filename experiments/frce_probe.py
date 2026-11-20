"""P5 France probe: the 0.975834 upload (v8.1 + ce2 stage 2 @0.64 + France rule P1) with ONLY the French decisions
changed: every French pair with prob >= 0.02 is re-scored by the same stage-2 model, but with the France-adapted
cross-encoder's score (kernel amizon-frce-w) in place of ce2. US/India rows stay byte-identical, so LB - 0.975834 is
the adapted CE's effect on France (x ~6.7 = on France's own score).

python experiments/frce_probe.py --frce <dir with test_ce.parquet> --out output_probe_frce
"""
import argparse
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # in the submission zip: src/experiments/ -> src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))  # git repo
from ce2_band import SECOND, logit  # noqa: E402
from run import decide, write  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frce", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=Path("output_probe_gc_rej"), help="the 0.975834 upload")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=float, default=0.64)
    a = ap.parse_args()
    # refit v8.1+ce2's stage-2 model exactly (reproduced the upload pair-for-pair on 27 Sep)
    ce = lambda f: pl.read_parquet(f).rename({"ce": "ce2"})  # noqa: E731
    B = Path("work/ce2band")
    tr_ce = pl.concat([ce(B / "h0/train_ce.parquet"), ce(B / "h1/train_ce.parquet")])
    o = pl.read_parquet("work/oof_v81.parquet", columns=["i", "j", "y", "oof"]).with_row_index("r")
    band = o.join(tr_ce, on=["i", "j"], how="inner").sort("r")
    m = lgb.LGBMClassifier(**SECOND, n_jobs=8).fit(
        np.column_stack([logit(band["oof"].to_numpy()), band["ce2"].to_numpy().astype(np.float32)]), band["y"].to_numpy())
    tp = pl.read_parquet("work/test_prob_v81ce2.parquet")
    fr = pl.scan_parquet("work9/test_s1/*.parquet").filter(pl.col("country") == "France").select(i="idx").collect()
    fc = pl.read_parquet(a.frce / "test_ce.parquet").rename({"ce": "cefr"})
    x = tp.join(fr, on="i").filter(pl.col("prob") >= 0.02).join(fc, on=["i", "j"], how="inner")
    n_fr = tp.join(fr, on="i").filter(pl.col("prob") >= 0.02).height
    assert x.height == n_fr, f"France-adapted scores cover {x.height} of {n_fr} French pairs"
    p2 = m.predict_proba(np.column_stack([logit(x["prob"].to_numpy()), x["cefr"].to_numpy().astype(np.float32)]))[:, 1]
    tp = tp.join(x.select("i", "j", p2=pl.Series(p2, dtype=pl.Float32)), on=["i", "j"], how="left").with_columns(
        prob=pl.coalesce("p2", "prob"))
    print(f"French pairs re-scored: {x.height}; accepted before/after (p >= {a.t}): "
          f"{(x['prob'] >= a.t).sum()} / {(p2 >= a.t).sum()}", flush=True)
    s1p, cp = "work/test_s1/*.parquet", "work/test_cand/*.parquet"
    q = decide(tp.select("i", "j"), tp["prob"].to_numpy(), a.t, True)
    tmp = Path(str(a.out) + "_pre")
    tmp.mkdir(parents=True, exist_ok=True)
    write(tmp / "matching_results.tsv", "matched_entity_ids", s1p, cp, q)
    (tmp / "candidate_pairs.tsv").write_bytes((a.base / "candidate_pairs.tsv").read_bytes())
    subprocess.run([sys.executable, "experiments/france_probe.py", "--base", str(tmp), "--groups", "work9/g_test_common.parquet",
                    "--mode", "reject", "--out", str(a.out)], check=True)
    # US/India rows must equal the 0.975834 upload
    rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False)  # noqa: E731
    b, n = rd(a.base / "matching_results.tsv"), rd(a.out / "matching_results.tsv")
    fr_ids = pl.read_csv("dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema=False) \
               .filter(pl.col("country") == "France").select(source1_entity_id="entity_id")
    diff = b.join(n, on="source1_entity_id").filter(pl.col("matched_entity_ids").fill_null("") != pl.col("matched_entity_ids_right").fill_null(""))
    assert diff.join(fr_ids, on="source1_entity_id", how="anti").height == 0, "a non-French row differs from the base"
    print(f"rows differing from the 0.975834 upload: {diff.height} (all French)", flush=True)


if __name__ == "__main__":
    main()
