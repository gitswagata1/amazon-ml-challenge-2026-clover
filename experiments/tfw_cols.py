"""Word-level TF-IDF cosines (tfw_core, tfw_addr) for v8's train and test pairs -> work/tfw_{train,test}.parquet.
Vectorisers are fitted on train + test text together (label-free), so French words (test only) get real IDF weights;
a train-only fit would treat every French word as unknown. Same settings as experiments/tfidf_ab.py (A/B: +0.0012).

python experiments/tfw_cols.py --work work --tag k30_p1500_ap_ak_use20
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

WORD = dict(analyzer="word", token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, min_df=2, dtype=np.float32)


def records(work, split, which):
    return (pl.scan_parquet(str(work / f"{split}_{which}" / "*.parquet")).select("idx", "core", "addr")
            .collect().sort("idx").with_columns(pl.col("core", "addr").fill_null("")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--tag", default="k30_p1500_ap_ak_use20", help="pairs file: {split}_pairs_<tag>.parquet")
    a = ap.parse_args()
    recs = {(s, w): records(a.work, s, w) for s in ("train", "test") for w in ("s1", "cand")}
    pairs = {s: pl.read_parquet(a.work / f"{s}_pairs_{a.tag}.parquet", columns=["i", "j"]) for s in ("train", "test")}
    out = {s: {"i": pairs[s]["i"], "j": pairs[s]["j"]} for s in pairs}
    for col in ("core", "addr"):
        vec = TfidfVectorizer(**WORD).fit(pl.concat([r[col] for r in recs.values()]).to_list())
        print(f"tfw_{col}: vocab {len(vec.vocabulary_)}", flush=True)
        for s, p in pairs.items():
            A, B = recs[(s, "s1")], recs[(s, "cand")]
            XA, XB = vec.transform(A[col].to_list()).tocsr(), vec.transform(B[col].to_list()).tocsr()
            ri = np.searchsorted(A["idx"].to_numpy(), p["i"].to_numpy())
            rj = np.searchsorted(B["idx"].to_numpy(), p["j"].to_numpy())
            cos = np.empty(p.height, np.float32)
            for o in range(0, p.height, 1_000_000):
                cos[o:o + 1_000_000] = np.asarray(XA[ri[o:o + 1_000_000]].multiply(XB[rj[o:o + 1_000_000]]).sum(1)).ravel()
            cos[(A[col].str.strip_chars() == "").to_numpy()[ri] | (B[col].str.strip_chars() == "").to_numpy()[rj]] = np.nan
            out[s][f"tfw_{col}"] = cos
            print(f"  {s}: {p.height} pairs, mean cos {np.nanmean(cos):.3f}", flush=True)
    for s in out:
        pl.DataFrame(out[s]).write_parquet(a.work / f"tfw_{s}.parquet")
    print("wrote work/tfw_train.parquet, work/tfw_test.parquet", flush=True)


if __name__ == "__main__":
    main()
