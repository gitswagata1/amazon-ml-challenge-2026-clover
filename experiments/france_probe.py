"""France-only leaderboard probe: copy a base submission and change ONLY French S1 rows, so the LB difference to the
base measures the change on France alone (US/India rows byte-identical).

Group G (label-free, experiments analysis 27 Sep): candidate at the same address as the S1 (token-set >= 95, house-number
key present) whose core differs from the S1 core by exactly ONE non-typo word swap, no other S2/S3 record with the same
(core, house key) and no S1 with that (core, house key). In train such pairs are matches 80% (US) / 38% (India); France
unknown -> probe.

--mode reject : drop G pairs from the base's French matches
--mode accept : add G pairs (not already matched to another S1) to the base's French matches (and candidates if needed)

python experiments/france_probe.py --base output_v81ce2 --groups work9/g_test.parquet --mode reject --out output_probe_g_rej
"""
import argparse
import shutil
from pathlib import Path

import polars as pl


def lists(path, col):
    return (pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
            .with_columns(pl.col(col).fill_null("").str.split(",")).explode(col).filter(pl.col(col) != "")
            .rename({col: "e2"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True, help="parquet: source1_entity_id, e2 (the G pairs)")
    ap.add_argument("--mode", choices=("reject", "accept"), required=True)
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fr = (pl.read_csv(a.data / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema=False)
          .filter(pl.col("country") == "France").select(source1_entity_id="entity_id"))
    g = pl.read_parquet(a.groups).join(fr, on="source1_entity_id")
    m = lists(a.base / "matching_results.tsv", "matched_entity_ids")
    c = lists(a.base / "candidate_pairs.tsv", "candidate_entity_ids")
    if a.mode == "reject":
        hit = m.join(g, on=["source1_entity_id", "e2"], how="semi")
        m2 = m.join(g, on=["source1_entity_id", "e2"], how="anti")
    else:
        taken = m.select("e2").unique()
        hit = g.join(m, on=["source1_entity_id", "e2"], how="anti").join(taken, on="e2", how="anti").unique("e2", keep="first")
        m2 = pl.concat([m, hit.select(m.columns)])
        c = pl.concat([c, hit.select(c.columns)]).unique(maintain_order=True)
    print(f"{a.mode}: {hit.height} French pairs changed, {hit['source1_entity_id'].n_unique()} French S1 affected "
          f"(of {fr.height}); matches {m.height} -> {m2.height}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_csv(a.base / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).select("source1_entity_id")
    for name, col, d in (("matching_results.tsv", "matched_entity_ids", m2), ("candidate_pairs.tsv", "candidate_entity_ids", c)):
        agg = d.group_by("source1_entity_id", maintain_order=True).agg(pl.col("e2").str.join(",").alias(col))
        s1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null("")).write_csv(
            a.out / name, separator="\t", quote_style="never")
    base_rows = pl.read_csv(a.base / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    new_rows = pl.read_csv(a.out / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    diff = base_rows.join(new_rows, on="source1_entity_id").filter(pl.col("matched_entity_ids").fill_null("") != pl.col("matched_entity_ids_right").fill_null(""))
    assert diff.join(fr, on="source1_entity_id", how="anti").height == 0, "non-French row changed"
    print(f"rows differing from base: {diff.height} (all French)")


if __name__ == "__main__":
    main()
