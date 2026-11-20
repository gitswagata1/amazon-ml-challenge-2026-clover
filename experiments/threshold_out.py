"""Leaderboard threshold probes without retraining: re-decide saved test probabilities (work/test_prob.parquet from
run.py predict) at another threshold, same one-owner rule, same candidate set.

python experiments/threshold_out.py --t 0.76 --out output_t076
"""
import argparse
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import decide, write  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--src", type=Path, default=Path("output"), help="predict output: candidate_pairs.tsv copied as is")
    a = ap.parse_args()
    s1p, cp = str(a.work / "test_s1" / "*.parquet"), str(a.work / "test_cand" / "*.parquet")
    p = pl.read_parquet(a.work / "test_prob.parquet")
    q = decide(p, p["prob"].to_numpy(), a.t, exclusive=True)
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "matching_results.tsv", "matched_entity_ids", s1p, cp, q)
    shutil.copy(a.src / "candidate_pairs.tsv", a.out / "candidate_pairs.tsv")
    print(f"t={a.t}: {q.height} matches, {q['i'].n_unique()} S1 with >=1 match -> {a.out}")


if __name__ == "__main__":
    main()
