"""Merge two validated submissions by country: rows (matching_results.tsv AND candidate_pairs.tsv) of the S1 records of
--countries come from --take, all other rows from --base. Each S1 row keeps the candidates its own model scored.
LB 27 Sep: v11+L1 (v10 base) 0.98533 -> with A's (v9 base) French rows 0.985926: v9 is better on France (+0.004),
v10 on US/India.

python experiments/country_merge.py --base output_final_v11L1 --take output_final_v9_control --countries France \
    --out output_probe_hyb_A_fr
"""
import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--take", type=Path, required=True)
    ap.add_argument("--countries", default="France")
    ap.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False)  # noqa: E731
    ids = (rd(a.test_dir / "test_source1.tsv").filter(pl.col("country").is_in(a.countries.split(",")))
           .select(source1_entity_id="entity_id"))
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        b, t = rd(a.base / f), rd(a.take / f)
        col = b.columns[1]
        assert b.columns == t.columns and b.height == t.height, f"{f}: files do not align"
        take = ids["source1_entity_id"].implode()
        m = (b.join(t.rename({col: "t"}), on="source1_entity_id", how="left", maintain_order="left")
             .with_columns(pl.when(pl.col("source1_entity_id").is_in(take)).then(pl.col("t")).otherwise(pl.col(col)).alias(col))
             .select(b.columns))
        d = m.join(b, on="source1_entity_id", suffix="_b").filter(pl.col(col).fill_null("") != pl.col(col + "_b").fill_null(""))
        assert d.join(ids, on="source1_entity_id", how="anti").height == 0, "a row outside --countries changed"
        print(f"{f}: {d.height} rows from {a.take} ({a.countries})", flush=True)
        m.write_csv(a.out / f, separator="\t", quote_style="never", null_value="")
    subprocess.run([sys.executable, "utils/validate_submission.py", "-m", str(a.out / "matching_results.tsv"),
                    "-c", str(a.out / "candidate_pairs.tsv"), "-t", str(a.test_dir), "--check-ids"], check=True)


if __name__ == "__main__":
    main()
