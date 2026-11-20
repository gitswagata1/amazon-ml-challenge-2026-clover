"""Partial A test: does the mined native-token dictionary improve the NAME FEATURES (blocking unchanged)?
A/B table (400k S1), tuned LightGBM, 5-fold by S1. Pairs where either name has non-ASCII text get their name
features recomputed with the current prep.py twice: without (control) and with the dictionary; nothing else differs.
The dictionary is mined from training S1 OUTSIDE the A/B table.

python experiments/a_dict_ab.py prep   # -> work/a_dict_{ctrl,dict}.parquet
python experiments/a_dict_ab.py cv
python experiments/a_dict_ab.py prep dictall && python experiments/a_dict_ab.py cv dictall   # v9-style dictionary (leakage size)
"""
import pickle
import re
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from features import _cp, _jacc
from prep import name_fields
from mine_dict import mine

D, NAT = "work/ab_data", r"[^\x00-\x7F]"
PUNCT = re.compile(r"[,()\[\]\-|&/]")


def fields(names, tr):
    """name, core, sk per raw name after optional token translation (same pre-tokenisation in both arms)."""
    out = [name_fields(" ".join(tr.get(t, t) for t in PUNCT.sub(" ", s.lower()).split())) for s in names]
    return [list(x) for x in zip(*out)]


def name_feats(A, B, vec):
    f = {}
    a, b = A[0], B[0]
    f["name_ratio"], f["name_partial"] = _cp(fuzz.ratio, a, b), _cp(fuzz.partial_ratio, a, b)
    f["name_tsort"], f["name_tset"] = _cp(fuzz.token_sort_ratio, a, b), _cp(fuzz.token_set_ratio, a, b)
    a, b = A[1], B[1]
    f["core_ratio"], f["core_tset"] = _cp(fuzz.ratio, a, b), _cp(fuzz.token_set_ratio, a, b)
    f["core_jw"] = _cp(JaroWinkler.normalized_similarity, a, b)
    ca, cb = [x.replace(" ", "") for x in a], [x.replace(" ", "") for x in b]
    f["compact_ratio"], f["compact_partial"] = _cp(fuzz.ratio, ca, cb), _cp(fuzz.partial_ratio, ca, cb)
    out = np.asarray(vec.transform(a).multiply(vec.transform(b)).sum(1), np.float32).ravel()
    out[[not x or not y for x, y in zip(a, b)]] = np.nan
    f["tfidf_core"] = out
    a, b = A[2], B[2]
    f["sk_ratio"], f["sk_tset"], f["sk_tsort"] = _cp(fuzz.ratio, a, b), _cp(fuzz.token_set_ratio, a, b), _cp(fuzz.token_sort_ratio, a, b)
    f["sk_jacc"] = _jacc(a, b, " ")
    f["sk_first_eq"] = np.array([x.split(" ")[0] == y.split(" ")[0] and x != "" for x, y in zip(a, b)], np.int8)
    f["sk_ntok_a"] = np.array([x.count(" ") + (x != "") for x in a], np.int8)
    f["sk_ntok_b"] = np.array([x.count(" ") + (x != "") for x in b], np.int8)
    f["name_len_a"] = np.array([len(x) for x in A[0]], np.int16)
    f["name_len_b"] = np.array([len(x) for x in B[0]], np.int16)
    return f


def prep_step():
    ab_s1 = pl.read_parquet(f"{D}/tune_meta.parquet")["i"]
    s1 = pl.scan_parquet("work/train_s1/*.parquet").select(i="idx", s1_name="business_name").collect()
    cand = pl.scan_parquet("work/train_cand/*.parquet").select(j="idx", cand_name="business_name").collect()
    truth = pl.read_parquet("work/train_truth.parquet").filter(~pl.col("i").is_in(ab_s1.implode()))
    d = mine(truth.join(s1, on="i").join(cand, on="j"))
    tr = dict(zip(d["tok"].to_list(), d["word"].to_list()))
    print(f"dictionary (mined outside the A/B S1): {d.height} tokens", flush=True)
    p = (pl.read_parquet(f"{D}/train_features.parquet", columns=["i", "j"]).join(s1, on="i").join(cand, on="j")
         .filter(pl.col("s1_name").str.contains(NAT) | pl.col("cand_name").str.contains(NAT)))
    print(f"pairs with non-ASCII names: {p.height}", flush=True)
    vec = pickle.load(open("work/tfidf.pkl", "rb"))["core"]
    arms = (("ctrl", {}), ("dict", tr))
    if len(sys.argv) > 2 and sys.argv[2] == "dictall":  # leakage check: dictionary mined on ALL training pairs (as in v9)
        import prep
        arms = (("dictall", prep.NATIVE),)
    for arm, t in arms:
        f = name_feats(fields(p["s1_name"].to_list(), t), fields(p["cand_name"].to_list(), t), vec)
        pl.DataFrame(f).with_columns(p["i"], p["j"]).write_parquet(f"work/a_dict_{arm}.parquet")
        print(f"{arm}: written", flush=True)


def cv_step():
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from ab import TUNED
    from tune import macro_f05
    base = (pl.read_parquet(f"{D}/train_features.parquet")
            .drop([c for c in ("r_rel_c", "r_gap_c", "sup_sk", "sup_hn", "sup_both") if c in pl.read_parquet_schema(f"{D}/train_features.parquet")])
            .join(pl.read_parquet(f"{D}/train_tfidf.parquet"), on=["i", "j"], how="left")
            .join(pl.read_parquet(f"{D}/train_emb_v3pairs.parquet"), on=["i", "j"], how="left").sort("i"))
    meta = pl.read_parquet(f"{D}/tune_meta.parquet")
    for arm in (sys.argv[2:] or ["ctrl", "dict"]):
        new = pl.read_parquet(f"work/a_dict_{arm}.parquet")
        cols = [c for c in new.columns if c not in ("i", "j")]
        df = (base.join(new, on=["i", "j"], how="left", suffix="_n")
              .with_columns([pl.coalesce(f"{c}_n", c).alias(c) for c in cols]).drop([f"{c}_n" for c in cols]))
        X, y, g = df.drop("i", "j", "y").to_numpy().astype(np.float32), df["y"].to_numpy(), df["i"].to_numpy()
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(X, y, g):
            oof[va] = lgb.LGBMClassifier(**TUNED).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        s, t = macro_f05(df, oof, meta)
        print(f"{arm}: CV macro F0.5 = {s:.5f} @ t={t}", flush=True)
        del df, X


if __name__ == "__main__":
    prep_step() if sys.argv[1] == "prep" else cv_step()
