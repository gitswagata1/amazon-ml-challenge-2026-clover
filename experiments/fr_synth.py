"""P5 (organiser-approved: self-training + synthetic pairs from the provided records): a French training set for
fine-tuning the ce2 cross-encoder on France, built only from the TEST French records (no labels, no external data).
  real positives   : French pairs the matcher is >= 99.5% sure of (one-owner winners), outside the France rules' lists
  sibling negatives: those positives with one generic French name word swapped for another (Club -> Comite; the
                     generator's look-alike recipe), same address (60%) or a shifted house number (40%)
  decoy negatives  : positive's name + a French decoy word (Groupe, Holding, ...) at a house number +1..15
  hard negatives   : P2' pairs (sibling-word swap at a different address; ~95% non-matches in simulation)
  easy negatives   : pairs the matcher scores < 0.01
Writes <out>/text/{train,test}_{s1,cand}.parquet (idx, business_name, business_address) and <out>/pairs/ce_train.parquet
(i, j, y) + test_pairs.parquet (French pairs to score) in ce.py's layout.

python experiments/fr_synth.py --out work9/frce
"""
import argparse
import random
import re
import unicodedata
from pathlib import Path

import polars as pl

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


def main():
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


if __name__ == "__main__":
    main()
