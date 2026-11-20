"""Load + normalise (cached as parquet), then key-based blocking with IDF-weighted meta-blocking.

Keys per record come from prep.block_keys (name skeleton combos + house numbers), all scoped by
country label, so unseen countries like France work unchanged.
Pairs sharing keys are scored by sum of idf(key) and the top-K candidates per S1 record are kept.
Everything streams from parquet so peak memory stays low.
"""
from pathlib import Path

import resource

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from prep import prep

BATCH = 400_000


def ensure(data: Path, work: Path, split: str, which: str) -> str:
    """which='s1' or 'cand' (S2+S3). Writes normalised parquet parts with a global idx (row number);
    returns a glob usable by pl.scan_parquet / read_parquet. Streams TSVs in batches to cap memory."""
    out = work / f"{split}_{which}"
    if not (out / "_done").exists():
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for k in [1] if which == "s1" else [2, 3]:
            reader = pl.read_csv_batched(data / split / f"{split}_source{k}.tsv", separator="\t", quote_char=None,
                                         infer_schema_length=0, batch_size=BATCH)
            while batch := reader.next_batches(1):
                b = prep(batch[0].fill_null("")).with_row_index("idx", offset=n)
                b.write_parquet(out / f"part-{n:09d}.parquet")
                n += b.height
        (out / "_done").touch()
    return str(out / "*.parquet")


def _keys(base: pl.DataFrame, id_col: str, addr_keys=False, raw=False) -> pl.DataFrame:
    """Keys from prep.block_keys plus address keys derived here:
    g:<first name token> <street>  (house number ignored) and
    d:<number variant> <street>    (number, number minus first digit, minus last digit: 1904~190, 3802~802).
    addr_keys: also a:<w1> <w2> for neighbouring non-numeric address words (order-free), so records with the
    same street/building but no shared name key or house number still meet (e.g. native-script names)."""
    bk = base.select(id_col, "ctry", k=pl.col("bk").str.split(" | ")).explode("k")
    hn = (base.select(id_col, "ctry", pl.col("sk").str.split(" ").list.first().alias("t"),
                      e=pl.col("hn").str.split(" | ")).explode("e")
          .filter(pl.col("e").str.contains(" "))
          .with_columns(n=pl.col("e").str.split(" ").list.first(), s=pl.col("e").str.split(" ").list.last()))
    g = hn.filter(pl.col("t").str.len_chars() >= 2).select(id_col, "ctry", k=pl.lit("g:") + pl.col("t") + " " + pl.col("s"))
    hn = hn.filter(pl.col("n").str.len_chars() >= 3)
    d = pl.concat([hn.select(id_col, "ctry", k=pl.lit("d:") + v + " " + pl.col("s"))
                   for v in (pl.col("n"), pl.col("n").str.slice(1), pl.col("n").str.slice(0, pl.col("n").str.len_chars() - 1))])
    parts = [bk, g, d]
    if addr_keys:
        w = (base.select(id_col, "ctry", w=pl.col("addr").str.split(" ").list.eval(
                pl.element().filter(~pl.element().str.contains(r"\d") & (pl.element().str.len_chars() >= 2))))
             .with_row_index("r").explode("w").drop_nulls("w").with_columns(pos=pl.int_range(pl.len()).over("r")))
        w = w.join(w.select("r", "pos", w2="w").with_columns(pl.col("pos") - 1), on=["r", "pos"])
        parts.append(w.select(id_col, "ctry", k=pl.lit("a:") + pl.min_horizontal("w", "w2") + " "
                              + pl.max_horizontal("w", "w2")))
    k = pl.concat(parts).filter(pl.col("k").str.len_chars() >= 3)
    if raw:  # unhashed key strings, for key analysis
        return k.select(id_col, "k").unique()
    return k.select(id_col, h=pl.concat_str("ctry", "k", separator="\x00").hash()).unique()


def keys(path: str, id_col: str, ctry: str, batch=1_000_000, addr_keys=False) -> pl.DataFrame:
    """-> DataFrame(id_col u32, h u64) of hashed (country, key) pairs. Batched to cap string memory."""
    lf = (pl.scan_parquet(path).filter(pl.col("ctry") == ctry)
          .select(pl.col("idx").alias(id_col), "ctry", "bk", "sk", "hn", "addr"))
    n = lf.select(pl.len()).collect().item()
    return pl.concat([_keys(lf.slice(o, batch).collect(), id_col, addr_keys) for o in range(0, max(n, 1), batch)])


def _rescore(q: pl.DataFrame, s1c: pl.DataFrame, cc: pl.DataFrame, addr_partial=False) -> pl.Series:
    """Cheap fuzzy re-rank score for pooled candidates: name skeleton + address token-set similarity
    (+ address partial ratio: top-20 recall 93.1% -> 94.1% at pool 1500)."""
    a = q.select("i").join(s1c, left_on="i", right_on="idx", how="left", maintain_order="left")
    b = q.select("j").join(cc, left_on="j", right_on="idx", how="left", maintain_order="left")
    name = process.cpdist(a["sk"].to_list(), b["sk"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    addr = process.cpdist(a["addr"].to_list(), b["addr"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    has = ((a["addr"] != "") & (b["addr"] != "")).to_numpy()
    hn = np.array([bool(set(x.split(" | ")) & set(y.split(" | "))) and x != ""
                   for x, y in zip(a["hn"].to_list(), b["hn"].to_list())])
    if addr_partial:
        addr = addr + process.cpdist(a["addr"].to_list(), b["addr"].to_list(), scorer=fuzz.partial_ratio, workers=-1)
    return pl.Series(name + np.where(has, addr, 50 * (1 + addr_partial)) + 40 * hn + 2 * q["bscore"].to_numpy(),
                     dtype=pl.Float32)



def block(s1_path: str, cand_path: str, out_path: Path, k=30, pool=600, max_df=10000, chunk=5_000, only=None,
          addr_keys=False, addr_partial=False):
    """Writes parquet(i, j, bscore, bshared, rscore, brank, ...): top-k candidates j per S1 row i.
    Stage 1: country-scoped key join, IDF-sum score, keep `pool` best. Stage 2: fuzzy re-rank, keep k.
    ponytail: exact country-label blocking (train shows 0 cross-country matches)."""
    parts = []
    for c in pl.scan_parquet(s1_path).select(pl.col("ctry").unique()).collect()["ctry"]:
        ck = keys(cand_path, "j", c, addr_keys=addr_keys)
        df = ck.group_by("h").len("df")
        n = df["df"].sum()
        df = df.filter(pl.col("df") <= max_df).select("h", w=(np.log(n) - pl.col("df").log()).cast(pl.Float32))
        ck = ck.join(df, on="h")  # (j, h, w)
        sk = keys(s1_path, "i", c, addr_keys=addr_keys).join(df.select("h"), on="h")
        if only is not None:
            sk = sk.filter(pl.col("i").is_in(only.implode()))
        del df
        cols = ["idx", "sk", "addr", "hn"]
        s1c = pl.scan_parquet(s1_path).filter(pl.col("ctry") == c).select(cols).collect()
        cc = pl.scan_parquet(cand_path).filter(pl.col("ctry") == c).select(cols).collect()
        ids, out = sk["i"].unique().sort(), []
        for lo in range(0, len(ids), chunk):
            part = ids.slice(lo, chunk)
            q = sk.filter(pl.col("i").is_between(part[0], part[-1])).join(ck, on="h")
            q = (q.group_by("i", "j").agg(bscore=pl.col("w").sum(), bshared=pl.len().cast(pl.UInt8))
                 .sort(["i", "bscore"], descending=[False, True]).group_by("i", maintain_order=True).head(pool))
            q = (q.with_columns(rscore=_rescore(q, s1c, cc, addr_partial))
                 .sort(["i", "rscore"], descending=[False, True]).group_by("i", maintain_order=True).head(k))
            out.append(q.with_columns(brank=pl.int_range(1, pl.len() + 1, dtype=pl.UInt16).over("i"),
                                      b_rel_s1=pl.col("bscore") / pl.col("bscore").max().over("i"),
                                      r_rel_s1=pl.col("rscore") / pl.col("rscore").max().over("i"),
                                      b_n_s1=pl.len().over("i").cast(pl.UInt16)))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        print(f"  blocked country={c!r}: {len(ids)} S1 with keys, peak RSS {rss} MB", flush=True)
        del ck, sk, s1c, cc
        if not out:  # no S1 records of this country to block (e.g. `only` filters them all out)
            continue
        parts.append(out_path.with_suffix(f".{len(parts)}.tmp"))
        pl.concat(out).write_parquet(parts[-1])
        del out
    pl.scan_parquet(parts).sink_parquet(out_path)
    for part in parts:
        part.unlink()


def cand_context(pairs_path: Path) -> pl.DataFrame:
    """Per candidate j: best blocking score over all S1 and how many S1 records picked it."""
    return (pl.scan_parquet(pairs_path).group_by("j")
            .agg(b_max_c=pl.col("bscore").max(), b_n_c=pl.len().cast(pl.UInt16)).collect(engine="streaming"))
