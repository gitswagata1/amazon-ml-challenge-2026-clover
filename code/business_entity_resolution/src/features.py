"""Pair features for (S1 row i, candidate row j). Country-agnostic, no hard-coded countries."""
import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

_POST = re.compile(r"\b\d{5,6}\b")


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def _jacc(a, b, sep):
    out = np.empty(len(a), np.float32)
    for n, (x, y) in enumerate(zip(a, b)):
        x, y = set(filter(None, x.split(sep))), set(filter(None, y.split(sep)))
        out[n] = len(x & y) / len(x | y) if x and y else -1
    return out


def _post(a):
    """Postcode = the LAST of the longest 5-6 digit numbers ('12345 main st austin tx 78701' -> '78701', not the
    house number); postcodes come after the street in US, India and France addresses."""
    m = _POST.findall(a)
    return max(reversed(m), key=len) if m else ""


COLS = ["name", "core", "sk", "addr", "hn", "entity_id"]


def _rows(path, ids: pl.Series):
    """Rows of the parquet at `path` aligned with ids (reads only the needed rows)."""
    rows = (pl.scan_parquet(path).select("idx", *COLS)
            .filter(pl.col("idx").is_in(ids.unique().implode())).collect())
    got = ids.to_frame("idx").join(rows, on="idx", how="left", maintain_order="left")
    return {c: got[c].to_list() for c in COLS}


def _tfidf_cos(vec, a, b):
    """Row-wise cosine of char-3-gram TF-IDF vectors (L2-normalised by the vectoriser); NaN if a side is empty."""
    out = np.asarray(vec.transform(a).multiply(vec.transform(b)).sum(1), np.float32).ravel()
    out[[not x or not y for x, y in zip(a, b)]] = np.nan
    return out


def features(p: pl.DataFrame, s1_path, cand_path, tfidf=None) -> pl.DataFrame:
    """p: blocking pairs joined with blocking.cand_context (optionally with emb_* columns from embed.py);
    tfidf: {"core": vectoriser, "addr": vectoriser} fitted on training text. Returns features aligned with p."""
    A, B = _rows(s1_path, p["i"]), _rows(cand_path, p["j"])
    f = {c: p[c].to_numpy() for c in ("bscore", "bshared", "rscore", "brank", "b_rel_s1", "r_rel_s1", "b_n_s1", "b_n_c")}
    f["b_rel_c"] = (p["bscore"] / p["b_max_c"]).to_numpy()
    a, b = A["name"], B["name"]
    f["name_ratio"] = _cp(fuzz.ratio, a, b)
    f["name_partial"] = _cp(fuzz.partial_ratio, a, b)
    f["name_tsort"] = _cp(fuzz.token_sort_ratio, a, b)
    f["name_tset"] = _cp(fuzz.token_set_ratio, a, b)
    a, b = A["core"], B["core"]
    f["core_ratio"] = _cp(fuzz.ratio, a, b)
    f["core_tset"] = _cp(fuzz.token_set_ratio, a, b)
    f["core_jw"] = _cp(JaroWinkler.normalized_similarity, a, b)
    ca, cb = [x.replace(" ", "") for x in a], [x.replace(" ", "") for x in b]
    f["compact_ratio"] = _cp(fuzz.ratio, ca, cb)
    f["compact_partial"] = _cp(fuzz.partial_ratio, ca, cb)
    a, b = A["sk"], B["sk"]
    f["sk_ratio"] = _cp(fuzz.ratio, a, b)
    f["sk_tset"] = _cp(fuzz.token_set_ratio, a, b)
    f["sk_tsort"] = _cp(fuzz.token_sort_ratio, a, b)
    f["sk_jacc"] = _jacc(a, b, " ")
    f["sk_first_eq"] = np.array([x.split(" ")[0] == y.split(" ")[0] and x != "" for x, y in zip(a, b)], np.int8)
    f["sk_ntok_a"] = np.array([x.count(" ") + (x != "") for x in a], np.int8)
    f["sk_ntok_b"] = np.array([x.count(" ") + (x != "") for x in b], np.int8)
    a, b = A["addr"], B["addr"]
    f["addr_ratio"] = _cp(fuzz.ratio, a, b)
    f["addr_tset"] = _cp(fuzz.token_set_ratio, a, b)
    f["addr_tsort"] = _cp(fuzz.token_sort_ratio, a, b)
    f["addr_partial"] = _cp(fuzz.partial_ratio, a, b)
    f["addr_len_a"] = np.array([len(x) for x in a], np.int16)
    f["addr_len_b"] = np.array([len(x) for x in b], np.int16)
    pa, pb = [_post(x) for x in a], [_post(x) for x in b]
    f["post_match"] = np.array([-1 if not x or not y else int(x == y) for x, y in zip(pa, pb)], np.int8)
    a, b = A["hn"], B["hn"]
    f["hn_jacc"] = _jacc(a, b, " | ")
    f["hn_num_jacc"] = _jacc([re.sub(r" [a-z]+", "", x) for x in a], [re.sub(r" [a-z]+", "", x) for x in b], " | ")
    f["name_len_a"] = np.array([len(x) for x in A["name"]], np.int16)
    f["name_len_b"] = np.array([len(x) for x in B["name"]], np.int16)
    f["src3"] = np.array([x.startswith("S3-") for x in B["entity_id"]], np.int8)
    for col, vec in (tfidf or {}).items():
        f[f"tfidf_{col}"] = _tfidf_cos(vec, A[col], B[col])
    for col in ("emb_name", "emb_addr", "ce"):  # GPU features: e5 cosine (embed.py), cross-encoder logit (ce.py)
        if col in p.columns:
            f[col] = p[col].to_numpy()
    for col in p.columns:  # kNN rank/cosine and per-split extra features (run.py --knn / --extra), passed through
        if col.startswith(("knn_", "tfw_", "rare_", "name_df", "cand_addr", "f1_", "f2_", "f3_", "f4_", "f5_", "f6_")):
            f[col] = p[col].cast(pl.Float32).to_numpy()
    return pl.DataFrame(f)
