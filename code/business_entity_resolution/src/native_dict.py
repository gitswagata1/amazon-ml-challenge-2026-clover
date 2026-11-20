"""Mine the native-script word dictionary used by prep.py from TRAINING match pairs only (no external data).

The data translate English business words into Indic scripts from a small fixed vocabulary (2,000 words cover 98.6%
of native-script tokens), e.g. குளோபல் = global, which letter-by-letter transliteration ('kulopal') cannot recover.
For every non-ASCII token of a matched candidate name, the dictionary keeps the S1-name word it co-occurs with most
(seen >= 5 times, in >= 50% of its pairs). Out-of-sample test: native-script misses matching word for word 0% -> 79%;
A/B name features +0.0010 CV.

python native_dict.py --work work --out code/business_entity_resolution/native_dict.tsv   (after blocking.ensure)
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl

NAT = r"[^\x00-\x7F]"
_toks = lambda e: e.str.to_lowercase().str.replace_all(r"[,()\[\]\-|&/]", " ").str.split(" ").list.eval(
    pl.element().filter(pl.element() != ""))


def mine(pairs: pl.DataFrame, min_n=5, min_share=0.5) -> pl.DataFrame:
    """pairs: s1_name, cand_name of true matches -> tok (non-ASCII candidate token), word (S1 word), n, share."""
    x = (pairs.with_columns(ct=_toks(pl.col("cand_name")), st=_toks(pl.col("s1_name")))
         .with_columns(st=pl.col("st").list.set_difference(pl.col("ct")))
         .select("ct", "st").with_row_index("r"))
    nat = x.select("r", "ct").explode("ct").filter(pl.col("ct").str.contains(NAT)).unique()
    co = nat.join(x.select("r", "st").explode("st").drop_nulls(), on="r").group_by("ct", "st").len("co")
    return (co.join(nat.group_by("ct").len("n"), on="ct").with_columns(share=pl.col("co") / pl.col("n"))
            .sort("co", "ct", "st", descending=[True, False, False]).unique("ct", keep="first", maintain_order=True)
            .filter((pl.col("n") >= min_n) & (pl.col("share") >= min_share))
            .select(tok="ct", word="st").sort("tok"))


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    import run
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "native_dict.tsv")
    a = ap.parse_args()
    s1p, cp = f"{a.work}/train_s1/*.parquet", f"{a.work}/train_cand/*.parquet"
    truth = run.truth_pairs(SimpleNamespace(data=a.data), s1p, cp)
    names = lambda p, c: pl.scan_parquet(p).select(pl.col("idx"), pl.col("business_name").alias(c)).collect()
    d = mine(truth.join(names(s1p, "s1_name"), left_on="i", right_on="idx")
             .join(names(cp, "cand_name"), left_on="j", right_on="idx"))
    d.write_csv(a.out, separator="\t")
    print(f"{d.height} tokens -> {a.out}")


if __name__ == "__main__":
    main()
