"""Write a submission from saved test probabilities with a separate threshold for one country (label-free choice, e.g.
France stricter per the v8 LB threshold probes: 0.68 0.973633 < 0.72 0.973867 < 0.76 0.973972). One-owner rule as run.py.

python experiments/country_threshold.py --work work9 --prob work9/test_prob_v9ce2.parquet --json output_v9ce2/ce2_band.json \
    --country France --offset 0.04 --out output_v9ce2_fr
Threshold and one-owner rule come from the stage's json (CV choice); the country gets threshold + offset (v8 LB probes:
+0.04 above the CV optimum was better, +0.000105). candidate_pairs.tsv = exactly the pairs in --prob (the scored set).
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import write  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--prob", type=Path, required=True, help="parquet i, j, prob for ALL test pairs")
    ap.add_argument("--json", type=Path, help="ce2_band.json / report json with threshold (+ exclusive)")
    ap.add_argument("--t", type=float, help="override the json threshold")
    ap.add_argument("--country", default="France")
    ap.add_argument("--offset", type=float, default=0.04, help="country threshold = t + offset")
    ap.add_argument("--conflict-delta", type=float, help="conflict rule: drop a record's winner when its 2nd-best S1 has "
                    "prob >= 0.5 and the gap is < delta (near-tie between look-alike S1; v9+ce2 CV +0.00018 @0.2)")
    ap.add_argument("--conflict-countries", default="US,India", help="countries the conflict rule applies to")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    js = json.loads(a.json.read_text()) if a.json else {}
    t = a.t if a.t is not None else js["threshold"]
    exclusive = js.get("exclusive", True)
    tc = round(t + a.offset, 4)
    s1p, cp = str(a.work / "test_s1" / "*.parquet"), str(a.work / "test_cand" / "*.parquet")
    c = pl.scan_parquet(s1p).select(i="idx", c="country").collect()
    allp = pl.read_parquet(a.prob, columns=["i", "j", "prob"])
    assert allp.height == allp.unique(["i", "j"]).height, "duplicate pairs in --prob"
    assert allp["prob"].null_count() == 0, "pairs without probability"
    p = allp.join(c, on="i")
    if a.conflict_delta is not None:  # 2nd-best prob of each record, before the one-owner filter
        second = (p.sort("prob", descending=True).with_columns(rk=pl.int_range(pl.len()).over("j"))
                  .filter(pl.col("rk") == 1).select("j", p2="prob"))
        p = p.join(second, on="j", how="left").with_columns(pl.col("p2").fill_null(0.0))
    if exclusive:  # one-owner rule
        p = p.filter(pl.col("prob") == pl.col("prob").max().over("j"))
    keep = pl.col("prob") >= pl.when(pl.col("c") == a.country).then(tc).otherwise(t)
    if a.conflict_delta is not None:
        near_tie = ((pl.col("p2") >= 0.5) & (pl.col("prob") - pl.col("p2") < a.conflict_delta)
                    & pl.col("c").is_in(a.conflict_countries.split(",")))
        print(f"conflict rule (delta {a.conflict_delta}, {a.conflict_countries}): "
              f"{p.filter(keep & near_tie).height} accepted winners dropped", flush=True)
        keep = keep & ~near_tie
    q = p.filter(keep).select("i", "j")
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "matching_results.tsv", "matched_entity_ids", s1p, cp, q)
    write(a.out / "candidate_pairs.tsv", "candidate_entity_ids", s1p, cp, allp.select("i", "j"))
    n = q.join(c, on="i").group_by("c").len().sort("c").rows()
    print(f"wrote {a.out}: {q.height} matches {n} (t={t}, exclusive={exclusive}, {a.country} t={tc}); "
          f"candidates = {allp.height} scored pairs", flush=True)


if __name__ == "__main__":
    main()
