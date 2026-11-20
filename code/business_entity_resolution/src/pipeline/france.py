"""France: training data for the domain-adapted cross-encoders, look-alike rules, and noise-variant recovery.

Subcommands (python pipeline/france.py <subcommand> --help):

  self-train-data
    Builds a French training set from the unlabelled French test records (text only, no labels): pseudo-labelled
    positives (pairs the pipeline scores >= 0.995 that are their record's best match), synthetic hard negatives that copy
    the data generator's look-alikes (one generic name word swapped, same or shifted house number; decoy words such as
    Groupe or Holding at a nearby house number), different-address look-alikes and easy negatives.
        python pipeline/france.py self-train-data --out work9/frce --score work9/test_prob_v9ce2.parquet

  rules
    Lists French look-alike pairs: the candidate's core name equals the Source 1 core name with one common word swapped
    (not a typo, not a generator noise word), at the same address with no other record or Source 1 of that name
    (default), or at a different address (--address different).
        python pipeline/france.py rules --work work9 --pairs work9/test_prob_v9.parquet --out work9/g_test_common.parquet

  apply
    Removes (--mode reject) or adds (--mode accept) a list of French pairs in a submission; other countries unchanged.
        python pipeline/france.py apply --base <submission dir> --groups <pair list> --mode reject --out <dir>

  recover
    Re-accepts French pairs the look-alike rule removed when the swapped word is a generator noise word (Services,
    Associés), and with --reverse-rules every removed pair that both French cross-encoders score >= 0.91.
        python pipeline/france.py recover --base <submission dir> --reverse-rules --out output
"""

import argparse
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import polars as pl
import numpy as np
from rapidfuzz import fuzz, process


W = Path("work9")
EXCLUDE = set("sci sarl sas sasu eurl sa ei snc selarl scp scop sca fils services service associes france groupe "
              "developpement holding participations international distribution de la le les du des et and the of".split())
DECOYS = ["Groupe", "France", "Développement", "Holding", "Participations", "International", "Distribution"]
NUM = re.compile(r"\d+")


def norm(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", t)


def style(word, like):
    return word.upper() if like.isupper() else (word.capitalize() if like[:1].isupper() else word.lower())


def swap_word(name, generic, rng):
    toks = name.split(" ")
    pos = [k for k, t in enumerate(toks) if norm(t) in generic]
    if not pos:
        return None
    k = rng.choice(pos)
    cur = norm(toks[k])
    new = rng.choice(generic)
    while new == cur:
        new = rng.choice(generic)
    toks[k] = style(new, toks[k])
    return " ".join(toks)


def shift_number(addr, rng):
    m = NUM.search(addr or "")
    if not m:
        return None
    return addr[:m.start()] + str(int(m.group()) + rng.randint(1, 15)) + addr[m.end():]


# ── self-train-data ────────────────────────────────────────────────────────────
def main_self_train_data():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prob", type=Path, default=W / "test_prob_v9ce2.parquet")
    ap.add_argument("--score", type=Path, action="append", default=[], help="test_prob parquet(s) whose French pairs "
                    "with prob >= 0.02 are written to test_pairs.parquet for scoring")
    ap.add_argument("--n-pos", type=int, default=300000)
    a = ap.parse_args()
    rng = random.Random(0)
    s1 = pl.scan_parquet(str(W / "test_s1" / "*.parquet")).filter(pl.col("country") == "France") \
           .select("idx", "business_name", "business_address", "core", "entity_id").collect()
    cd = pl.scan_parquet(str(W / "test_cand" / "*.parquet")).filter(pl.col("country") == "France") \
           .select("idx", "business_name", "business_address", "entity_id").collect()
    # generic French name words: frequent in French S1 cores, not city names (frequent address tokens), not noise/legal
    df = s1.select(w=pl.col("core").str.split(" ")).explode("w").group_by("w").len("df").filter(pl.col("df") >= 200)
    cities = set(s1.select(w=pl.col("business_address").str.to_lowercase().str.extract_all(r"[a-z]+")).explode("w")
                 .group_by("w").len("n").filter(pl.col("n") >= 2000)["w"].to_list())
    generic = sorted(w for w in df["w"].to_list() if w and w not in EXCLUDE and w not in cities and len(w) >= 3)
    print(f"{len(generic)} generic words, e.g. {generic[:25]}", flush=True)
    gset = set(generic)

    p = pl.read_parquet(a.prob).join(s1.select(i="idx"), on="i")
    p = p.with_columns(win=pl.col("prob") == pl.col("prob").max().over("j"))
    ids = lambda f: pl.read_parquet(f).join(s1.select(source1_entity_id="entity_id", i="idx"), on="source1_entity_id") \
        .join(cd.select(e2="entity_id", j="idx"), on="e2").select("i", "j")
    rules = pl.concat([ids(W / "g_test_common.parquet"), ids(W / "g2_test_diffaddr_sym.parquet")]).unique()
    pos = p.filter((pl.col("prob") >= 0.995) & pl.col("win")).join(rules, on=["i", "j"], how="anti") \
           .sample(a.n_pos, seed=0).select("i", "j")
    easy = p.filter(pl.col("prob") < 0.01).sample(a.n_pos // 3, seed=1).select("i", "j")
    hard = ids(W / "g2_test_diffaddr_sym.parquet").join(p.select("i", "j"), on=["i", "j"]).sample(
        min(a.n_pos // 2, 263000), seed=2)
    t1 = dict(zip(s1["idx"], zip(s1["business_name"].fill_null(""), s1["business_address"].fill_null(""))))
    t2 = dict(zip(cd["idx"], zip(cd["business_name"].fill_null(""), cd["business_address"].fill_null(""))))

    rows = []  # (s1 name, s1 addr, cand name, cand addr, y)
    for i, j in pos.iter_rows():
        rows.append((*t1[i], *t2[j], 1))
    for i, j in pos.sample(fraction=1.0, shuffle=True, seed=3).iter_rows():   # sibling negatives
        n2, a2 = t2[j]
        s = swap_word(n2, generic, rng)
        if s is None:
            continue
        if rng.random() < 0.4:
            a2 = shift_number(a2, rng) or a2
        rows.append((*t1[i], s, a2, 0))
    for i, j in pos.sample(n=a.n_pos // 3, seed=4).iter_rows():             # decoys
        n2, a2 = t2[j]
        d = rng.choice(DECOYS)
        a2s = shift_number(a2, rng)
        if a2s is None:
            continue
        rows.append((*t1[i], f"{n2} {style(d, n2.split(' ')[-1] if n2 else d)}", a2s, 0))
    for i, j in pl.concat([hard, easy]).iter_rows():
        rows.append((*t1[i], *t2[j], 0))
    rng.shuffle(rows)
    y = [r[4] for r in rows]
    print(f"{len(rows)} training pairs, positives {sum(y)} ({sum(y) / len(rows):.1%})", flush=True)

    (a.out / "text").mkdir(parents=True, exist_ok=True)
    (a.out / "pairs").mkdir(parents=True, exist_ok=True)
    n = len(rows)
    pl.DataFrame({"idx": range(n), "business_name": [r[0] for r in rows], "business_address": [r[1] for r in rows]},
                 schema_overrides={"idx": pl.UInt32}).write_parquet(a.out / "text" / "train_s1.parquet")
    pl.DataFrame({"idx": range(n), "business_name": [r[2] for r in rows], "business_address": [r[3] for r in rows]},
                 schema_overrides={"idx": pl.UInt32}).write_parquet(a.out / "text" / "train_cand.parquet")
    pl.DataFrame({"i": range(n), "j": range(n), "y": y}, schema_overrides={"i": pl.UInt32, "j": pl.UInt32, "y": pl.Int8}) \
      .write_parquet(a.out / "pairs" / "ce_train.parquet")
    s1.select("idx", "business_name", "business_address").write_parquet(a.out / "text" / "test_s1.parquet")
    cd.select("idx", "business_name", "business_address").write_parquet(a.out / "text" / "test_cand.parquet")
    sc = pl.concat([pl.read_parquet(f, columns=["i", "j", "prob"]).filter(pl.col("prob") >= 0.02).select("i", "j")
                    for f in (a.score or [a.prob])]).unique().join(s1.select(i="idx"), on="i")
    sc.write_parquet(a.out / "pairs" / "test_pairs.parquet")
    print(f"wrote {a.out}: test pairs to score {sc.height}", flush=True)


# ── rules ──────────────────────────────────────────────────────────────────────
def main_rules():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="parquet with i, j = all scored test pairs")
    ap.add_argument("--country", default="France", help="country to output, or ALL")
    ap.add_argument("--split", default="test", help="test (default) or train (for simulations)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr-max", type=float, help="max log add/remove ratio of the added word (default: 2 for same address, 0.5 for different)")
    ap.add_argument("--address", choices=("same", "different"), default="same",
                    help="same = same-address look-alikes; different = the same swap at a different address (twin checks skipped)")
    a = ap.parse_args()
    if a.lr_max is None:
        a.lr_max = 2.0 if a.address == "same" else 0.5
    s1 = pl.scan_parquet(str(a.work / f"{a.split}_s1" / "*.parquet")).select(i="idx", c="country", n1="core", a1="addr", h1="hn",
                                                                       s="entity_id").collect()
    cd = pl.scan_parquet(str(a.work / f"{a.split}_cand" / "*.parquet")).select(j="idx", c2="country", n2="core", a2="addr", h2="hn",
                                                                         e2="entity_id").collect()
    d = pl.read_parquet(a.pairs, columns=["i", "j"]).join(s1, on="i").join(cd, on="j")
    d = d.with_columns(t1=pl.col("n1").str.split(" ").list.unique(), t2=pl.col("n2").str.split(" ").list.unique())
    d = d.with_columns(ra=pl.col("t1").list.set_difference("t2"), rb=pl.col("t2").list.set_difference("t1"),
                       k=pl.col("t1").list.set_intersection("t2").list.len())
    d = d.filter((pl.col("ra").list.len() == 1) & (pl.col("rb").list.len() == 1) & (pl.col("k") >= 1)).with_columns(
        rem=pl.col("ra").list.first(), add=pl.col("rb").list.first()).drop("t1", "t2", "ra", "rb")
    d = d.with_columns(tsim=pl.Series(process.cpdist(d["rem"].to_list(), d["add"].to_list(), scorer=fuzz.ratio, workers=-1))) \
         .filter(pl.col("tsim") < 70)
    m = (d.group_by("c", w="add").len("n_add").join(d.group_by("c", w="rem").len("n_rem"), on=["c", "w"], how="full", coalesce=True)
         .fill_null(0).with_columns(lr=((pl.col("n_add") + 5) / (pl.col("n_rem") + 5)).log()))
    d = d.join(m.select("c", add="w", lr_add="lr"), on=["c", "add"], how="left").filter(pl.col("lr_add") <= a.lr_max)
    wdf = s1.with_columns(w=pl.col("n1").str.split(" ").list.unique()).explode("w").group_by("c", "w").len("df")
    d = (d.join(wdf.rename({"w": "add", "df": "df_add"}), on=["c", "add"], how="left")
         .join(wdf.rename({"w": "rem", "df": "df_rem"}), on=["c", "rem"], how="left").fill_null(0)
         .filter((pl.col("df_add") >= 20) & (pl.col("df_rem") >= 20)))
    if a.address == "different":
        d = d.with_columns(asim=pl.Series(process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
        d = d.filter((pl.col("a2") != "") & (pl.col("asim") < 95))  # a real, different address
        print("different-address look-alike pairs per country:", d.group_by("c").len().sort("c").rows(), flush=True)
        out = (d if a.country == "ALL" else d.filter(pl.col("c") == a.country)).select("i", "j", "c", source1_entity_id="s", e2="e2")
        out.write_parquet(a.out)
        print(f"wrote {a.out}: {out.height} {a.country} pairs", flush=True)
        return
    d = d.filter(pl.col("h2") != "")
    nrec = cd.filter(pl.col("h2") != "").group_by("c2", "n2", "h2").len("nrec")
    ns1 = s1.filter(pl.col("h1") != "").group_by(c2="c", n2="n1", h2="h1").len("ns1")
    d = d.join(nrec, on=["c2", "n2", "h2"], how="left").join(ns1, on=["c2", "n2", "h2"], how="left").fill_null(0) \
         .filter((pl.col("nrec") <= 1) & (pl.col("ns1") == 0))
    d = d.with_columns(asim=pl.Series(process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    d = d.filter(pl.col("asim") >= 95)
    print("G-common pairs per country:", d.group_by("c").len().sort("c").rows(), flush=True)
    out = (d if a.country == "ALL" else d.filter(pl.col("c") == a.country)).select("i", "j", "c", source1_entity_id="s", e2="e2")
    out.write_parquet(a.out)
    print(f"wrote {a.out}: {out.height} {a.country} pairs", flush=True)


def lists(path, col):
    return (pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
            .with_columns(pl.col(col).fill_null("").str.split(",")).explode(col).filter(pl.col(col) != "")
            .rename({col: "e2"}))


# ── apply ─────────────────────────────────────────────────────────────────────
def main_apply():
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




def swaps(prob):
    """French same-address single-word swaps (one word removed, one added, >= 1 shared, not typo-like) with house keys."""
    s1 = pl.scan_parquet(str(W / "test_s1" / "*.parquet")).filter(pl.col("country") == "France") \
           .select(i="idx", n1="core", a1="addr", h1="hn", source1_entity_id="entity_id").collect()
    cd = pl.scan_parquet(str(W / "test_cand" / "*.parquet")).filter(pl.col("country") == "France") \
           .select(j="idx", n2="core", a2="addr", h2="hn", e2="entity_id").collect()
    d = pl.read_parquet(prob, columns=["i", "j", "prob"]).join(s1, on="i").join(cd, on="j")
    d = d.with_columns(t1=pl.col("n1").str.split(" ").list.unique(), t2=pl.col("n2").str.split(" ").list.unique())
    d = d.with_columns(ra=pl.col("t1").list.set_difference("t2"), rb=pl.col("t2").list.set_difference("t1"),
                       k=pl.col("t1").list.set_intersection("t2").list.len())
    d = d.filter((pl.col("ra").list.len() == 1) & (pl.col("rb").list.len() == 1) & (pl.col("k") >= 1)) \
         .with_columns(rem=pl.col("ra").list.first(), add=pl.col("rb").list.first())
    d = d.with_columns(tsim=pl.Series(process.cpdist(d["rem"].to_list(), d["add"].to_list(), scorer=fuzz.ratio, workers=-1))) \
         .filter(pl.col("tsim") < 70)
    d = d.filter((pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")))  # same house number
    nrec = cd.filter(pl.col("h2") != "").group_by("n2", "h2").len("nrec")
    ns1 = s1.filter(pl.col("h1") != "").group_by(n2="n1", h2="h1").len("ns1")
    d = d.join(nrec, on=["n2", "h2"], how="left").join(ns1, on=["n2", "h2"], how="left").fill_null(0)
    d = d.filter((pl.col("nrec") <= 1) & (pl.col("ns1") == 0))  # single record, no S1 with the record's name there
    d = d.with_columns(asim=pl.Series(process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    return d.filter(pl.col("asim") >= 95).select("source1_entity_id", "e2", "add", "prob")


# ── recover ─────────────────────────────────────────────────────────────────
def main_recover():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--prob", type=Path, default=W / "test_prob_v9_p5b.parquet", help="French stage-2 probabilities (with the French cross-encoders)")
    ap.add_argument("--p1", type=Path, default=W / "g_test_common.parquet", help="the same-address look-alike list (source1_entity_id, e2)")
    ap.add_argument("--words", default="", help="extra look-alike-rule words to re-accept at prob >= 0.91 (comma list)")
    ap.add_argument("--reverse-rules", action="store_true",
                    help="second pass: re-accept every pair removed by the look-alike rules that is its record's best match "
                         "winner with prob >= 0.91 on a record no S1 owns (both France cross-encoders were confident)")
    ap.add_argument("--p2", type=Path, default=W / "g2_test_diffaddr_sym.parquet", help="the different-address look-alike list")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = swaps(a.prob)
    p1 = pl.read_parquet(a.p1).select("source1_entity_id", "e2").unique()
    sel = d.filter((pl.col("add") == "associes") | ((pl.col("add") == "services") & (pl.col("prob") >= 0.5)))
    extra = [w for w in a.words.split(",") if w]
    if extra:
        sel = pl.concat([sel, d.filter(pl.col("add").is_in(extra) & (pl.col("prob") >= 0.91)).join(p1, on=["source1_entity_id", "e2"], how="semi")])
    g = sel.sort("prob", descending=True).unique("e2", keep="first").select("source1_entity_id", "e2")
    gp = Path(f"{a.out}_accept.parquet")
    g.write_parquet(gp)
    print(f"noise-copy candidates: {g.height} ({sel.group_by('add').len().sort('len', descending=True).rows()})", flush=True)
    first = Path(f"{a.out}_noise") if a.reverse_rules else a.out
    subprocess.run([sys.executable, __file__, "apply", "--base", str(a.base), "--groups", str(gp), "--mode", "accept",
                    "--out", str(first)], check=True)
    if a.reverse_rules:
        s1 = pl.scan_parquet(str(W / "test_s1" / "*.parquet")).select(i="idx", source1_entity_id="entity_id").collect()
        cd = pl.scan_parquet(str(W / "test_cand" / "*.parquet")).select(j="idx", e2="entity_id").collect()
        p = pl.read_parquet(a.prob, columns=["i", "j", "prob"])
        win = p.filter(pl.col("prob") == pl.col("prob").max().over("j")).select("i", "j")
        rules = pl.concat([pl.read_parquet(f).select("source1_entity_id", "e2") for f in (a.p1, a.p2)]).unique() \
                  .join(s1, on="source1_entity_id").join(cd, on="e2").join(p, on=["i", "j"])
        rev = rules.filter(pl.col("prob") >= 0.91).join(win, on=["i", "j"], how="semi") \
                   .sort("prob", descending=True).unique("e2", keep="first").select("source1_entity_id", "e2")
        rp = Path(f"{a.out}_reverse.parquet")
        rev.write_parquet(rp)
        print(f"rule reversal candidates (winners, prob >= 0.91): {rev.height}", flush=True)
        subprocess.run([sys.executable, __file__, "apply", "--base", str(first), "--groups", str(rp), "--mode", "accept",
                        "--out", str(a.out)], check=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "-m", str(a.out / "matching_results.tsv"),
                    "-c", str(a.out / "candidate_pairs.tsv"), "-t", "dataset/test", "--check-ids"], check=True)


SUBCOMMANDS = {"self-train-data": main_self_train_data, "rules": main_rules, "apply": main_apply, "recover": main_recover}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        sys.exit("usage: python france.py {" + " | ".join(SUBCOMMANDS) + "} [options]")
    SUBCOMMANDS[sys.argv.pop(1)]()
