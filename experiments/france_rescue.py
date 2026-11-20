"""France recall rescue (27 Sep, last hour): French candidate pairs the US/India-trained stage 1 scored < 0.02, on records
no S1 owns in the base file, re-scored by the France-adapted CE P5 (Kaggle shards rescue_out_{w,s,g}/test_ce.parquet).
For each such record: its best S1 by P5; accept the pair if the P5 logit >= --tau. Added with france_probe.py --mode accept
(one owner per record; candidate lists unchanged: these pairs are already scored candidates).

python experiments/france_rescue.py --tau 9 --out output_probe_rescue9
"""
import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output_probe_hyb2_p5b_frt0.20", help="best file (LB 0.986368)")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    parts = [pl.read_parquet(f) for f in sorted(Path("work9").glob("rescue_out_*/test_ce.parquet"))]
    assert parts, "no rescue scores yet"
    sc = pl.concat(parts).unique(["i", "j"])
    best = sc.sort("ce", descending=True).unique("j", keep="first")
    acc = best.filter(pl.col("ce") >= a.tau)
    s1 = pl.scan_parquet("work9/test_s1/*.parquet").select(i="idx", source1_entity_id="entity_id").collect()
    cd = pl.scan_parquet("work9/test_cand/*.parquet").select(j="idx", e2="entity_id").collect()
    g = acc.join(s1, on="i").join(cd, on="j").select("source1_entity_id", "e2")
    gp = Path(f"work9/rescue_groups_{a.tau}.parquet")
    g.write_parquet(gp)
    print(f"rescue scores {sc.height} pairs ({len(parts)} shards), records {best.height}; accepted at logit >= {a.tau}: {acc.height} "
          f"pairs in {acc['i'].n_unique()} S1", flush=True)
    subprocess.run([sys.executable, "experiments/france_probe.py", "--base", a.base, "--groups", str(gp), "--mode", "accept",
                    "--out", str(a.out)], check=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "-m", str(a.out / "matching_results.tsv"),
                    "-c", str(a.out / "candidate_pairs.tsv"), "-t", "dataset/test", "--check-ids"], check=True)


if __name__ == "__main__":
    main()
