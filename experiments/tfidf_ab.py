"""TF-IDF variants for the shared A/B table (experiments/ab.py --extra ...), beyond the char-3-gram core/addr cosines
already in the model. Each record is vectorised once, then row-wise cosines per (i, j) pair; empty side -> NaN.
Vectorisers are fitted on training text only (the table's own S1 + candidate records).

python experiments/tfidf_ab.py --dir ab_data --work work      # -> ab_data/extra.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

WORD = dict(analyzer="word", token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, min_df=2, dtype=np.float32)
VARIANTS = {  # name: (text column, vectoriser settings)
    "tfw_core": ("core", WORD),                       # word-level: rare words weigh more than 'traders'
    "tfw_addr": ("addr", WORD),
    "tfc24_name": ("name", dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2, dtype=np.float32)),
    "tfc3_all": ("all", dict(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, dtype=np.float32)),
}


def texts(work, which, ids):
    t = (pl.scan_parquet(str(work / f"train_{which}" / "*.parquet")).select("idx", "name", "core", "addr")
         .filter(pl.col("idx").is_in(ids.implode())).collect().sort("idx"))
    return t.with_columns(all=pl.col("name") + " " + pl.col("addr"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("work"))
    a = ap.parse_args()
    p = pl.read_parquet(a.dir / "train_features.parquet", columns=["i", "j"])
    A, B = texts(a.work, "s1", p["i"].unique()), texts(a.work, "cand", p["j"].unique())
    ri = np.searchsorted(A["idx"].to_numpy(), p["i"].to_numpy())
    rj = np.searchsorted(B["idx"].to_numpy(), p["j"].to_numpy())
    out = {"i": p["i"], "j": p["j"]}
    for name, (col, kw) in VARIANTS.items():
        a_txt, b_txt = A[col].fill_null("").to_list(), B[col].fill_null("").to_list()
        vec = TfidfVectorizer(**kw).fit(a_txt + b_txt)
        XA, XB = vec.transform(a_txt).tocsr(), vec.transform(b_txt).tocsr()
        cos = np.empty(p.height, np.float32)
        for s in range(0, p.height, 1_000_000):
            cos[s:s + 1_000_000] = np.asarray(XA[ri[s:s + 1_000_000]].multiply(XB[rj[s:s + 1_000_000]]).sum(1)).ravel()
        empty = (A[col].fill_null("").str.strip_chars() == "").to_numpy()[ri] | \
                (B[col].fill_null("").str.strip_chars() == "").to_numpy()[rj]
        cos[empty] = np.nan
        out[name] = cos
        print(f"{name}: vocab {len(vec.vocabulary_)}, mean cos {np.nanmean(cos):.3f}", flush=True)
    pl.DataFrame(out).write_parquet(a.dir / "extra.parquet")
    print(f"wrote {a.dir / 'extra.parquet'} ({p.height} pairs)", flush=True)


if __name__ == "__main__":
    main()
