"""Mine a non-ASCII token -> Latin word dictionary from TRAINING match pairs (no external data), and measure how
many v8-missed native-script pairs get the same normalised name once candidates are translated with it.
Mined on S1 records of ce_half 0, evaluated on half 1 (so the check is out-of-sample).

python experiments/mine_dict.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from ce_data import ce_half  # noqa: E402

NAT = r"[^\x00-\x7F]"
toks = lambda e: e.str.to_lowercase().str.replace_all(r"[,()\[\]\-|&/]", " ").str.split(" ").list.eval(
    pl.element().filter(pl.element() != ""))


def mine(pairs, min_n=5, min_share=0.5):
    """pairs: i, j, s1_name, cand_name -> native token -> most co-occurring S1 token not already in the cand name."""
    x = (pairs.with_columns(ct=toks(pl.col("cand_name")), st=toks(pl.col("s1_name")))
         .with_columns(st=pl.col("st").list.set_difference(pl.col("ct")))
         .select("ct", "st").with_row_index("r"))
    nat = x.select("r", "ct").explode("ct").filter(pl.col("ct").str.contains(NAT)).unique()
    co = nat.join(x.select("r", "st").explode("st").drop_nulls(), on="r").group_by("ct", "st").len("co")
    n = nat.group_by("ct").len("n")
    return (co.join(n, on="ct").with_columns(share=pl.col("co") / pl.col("n"))
            .sort("co", descending=True).unique("ct", keep="first")
            .filter((pl.col("n") >= min_n) & (pl.col("share") >= min_share)).select("n", "share", tok="ct", word="st"))


def main():
    s1 = pl.scan_parquet("work/train_s1/*.parquet").select(i="idx", s1_name="business_name", ctry="ctry").collect()
    cand = pl.scan_parquet("work/train_cand/*.parquet").select(j="idx", cand_name="business_name").collect()
    truth = pl.read_parquet("work/train_truth.parquet").join(s1, on="i").join(cand, on="j")
    d = mine(truth.filter(ce_half(pl.col("i")) == 0))
    d.write_parquet("work/native_dict.parquet")
    print(f"dictionary: {d.height} tokens")
    print(d.sort("n", descending=True).head(25))

    # out-of-sample check on half 1: v8-missed pairs whose candidate name has native tokens
    tr = dict(zip(d["tok"].to_list(), d["word"].to_list()))
    m = (pl.read_parquet("work/v8_missed/v8_missed.parquet").filter(ce_half(pl.col("i")) == 1)
         .join(s1, on="i").join(cand, on="j").filter(pl.col("cand_name").str.contains(NAT)))
    norm = lambda s: " ".join(sorted(set(s)))
    same0 = same1 = 0
    for a, b in zip(m["s1_name"].to_list(), m["cand_name"].to_list()):
        ta = [t for t in a.lower().replace(",", " ").split() if t]
        tb = [t for t in b.lower().replace(",", " ").split() if t]
        same0 += set(tb) <= set(ta)
        same1 += {tr.get(t, t) for t in tb} <= set(ta)
    print(f"half-1 v8 misses with native tokens: {m.height}; candidate words all found in the S1 name: "
          f"before {same0 / m.height:.3f}, after translation {same1 / m.height:.3f}")
    left = [(a, b) for a, b in zip(m["s1_name"].to_list(), m["cand_name"].to_list())
            if not {tr.get(t, t) for t in b.lower().replace(",", " ").split()} <= set(a.lower().replace(",", " ").split())]
    for a, b in left[:12]:
        print("  still differs:", a, "|", b, "->", " ".join(tr.get(t, t) for t in b.lower().split()))


if __name__ == "__main__":
    main()
