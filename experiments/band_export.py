"""Export a matcher's uncertain band for the ce2 (e5-base, out-of-fold halves) score-only kernels.
Band = pairs with probability in [0.02, 0.98]: train from the OOF predictions, test from the test predictions.
Model h0 (trained on half-0 S1) scores train rows of half 1, h1 scores half 0; both score the whole test band.
-> <out>/h0/{train,test}_pairs.parquet, <out>/h1/{train,test}_pairs.parquet (i, j), one Kaggle dataset per half.

python experiments/band_export.py --oof work9/oof_v9.parquet --test-prob work9/test_prob_v9.parquet --out work9/band
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # in the submission zip: src/experiments/ -> src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))  # git repo
from ce_data import ce_half  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--oof", type=Path, required=True, help="i, j, y, oof")
ap.add_argument("--test-prob", type=Path, required=True, help="i, j, prob")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
band = lambda f, c: pl.read_parquet(f, columns=["i", "j", c]).filter(pl.col(c).is_between(0.02, 0.98)).select("i", "j")
tr, te = band(a.oof, "oof"), band(a.test_prob, "prob")
for h in (0, 1):
    d = a.out / f"h{h}"
    d.mkdir(parents=True, exist_ok=True)
    tr.filter(ce_half(pl.col("i")) == 1 - h).write_parquet(d / "train_pairs.parquet")
    te.write_parquet(d / "test_pairs.parquet")
    print(f"h{h}: train {tr.filter(ce_half(pl.col('i')) == 1 - h).height}, test {te.height}")
