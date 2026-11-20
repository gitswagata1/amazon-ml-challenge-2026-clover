"""A miniature version of the whole competition pipeline that runs on a laptop in under a minute.

It fakes a small dataset with the same kinds of noise as the real one, then does exactly what the real
pipeline does, in the same order:

    1. normalise text            (real: src/prep.py)
    2. block -> candidates       (real: src/blocking.py, IDF-weighted keys)
    3. pair features             (real: src/features.py)
    4. classifier + group CV     (real: src/run.py, pipeline/classifier.py, LightGBM)
    5. one-to-one + threshold    (real: run.decide, pipeline/decide.py)
    6. score with macro F0.5     (real: run.f05)

    pip install numpy scikit-learn
    python examples/toy_entity_resolution.py

Read it top to bottom; every step prints what it did. Change the numbers at the top and re-run to build
intuition (e.g. set K = 2 and watch candidate recall, and therefore the final score, fall).
"""
import math
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

N_BUSINESSES = 3000   # Source 1 records
K = 10                # candidates kept per Source 1 record after blocking
SEED = 7
rng = random.Random(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Fake data with realistic noise
# ─────────────────────────────────────────────────────────────────────────────
FIRST = ["Sharma", "Global", "Royal", "Sunrise", "Patel", "Smith", "Blue", "Star", "Lakshmi", "Eagle", "Metro",
         "Prime", "Green", "Apex", "Silver", "Ganesh", "Liberty", "Summit", "Vision", "Kumar"]
TRADE = ["Traders", "Software", "Builders", "Textiles", "Foods", "Logistics", "Motors", "Pharma", "Electricals",
         "Consulting", "Hardware", "Steel", "Plastics", "Exports", "Media"]
GENERIC = ["Services", "Solutions", "Enterprises", "Group", "Partners", "Center"]
LEGAL = {"US": ["LLC", "Inc", "Corp", ""], "India": ["Pvt Ltd", "Private Limited", "LLP", ""]}
STREET = {"US": ["Main Street", "Oak Avenue", "Park Road", "Lake Drive", "Elm Street"],
          "India": ["MG Road", "Station Road", "Nehru Nagar", "Gandhi Marg", "Temple Street"]}
CITY = {"US": ["Austin", "Denver", "Boston", "Dallas"], "India": ["Pune", "Chennai", "Indore", "Mysore"]}
ABBR = {"Street": "St", "Road": "Rd", "Avenue": "Ave", "Drive": "Dr", "Private": "Pvt", "Limited": "Ltd",
        "Corporation": "Corp", "Services": "Svcs"}


def typo(s):
    if len(s) < 5:
        return s
    k = rng.randrange(1, len(s) - 1)
    return s[:k] + s[k + 1] + s[k] + s[k + 2:]           # swap two letters


def noisy_copy(name, addr):
    """A Source 2/3 record of the SAME business: abbreviations, typos, dropped legal form, dropped city."""
    words = name.split()
    if rng.random() < 0.4:
        words = [ABBR.get(w, w) for w in words]
    if rng.random() < 0.3 and len(words) > 2:
        words = words[:2]                                # legal form dropped
    if rng.random() < 0.25:
        k = rng.randrange(len(words))
        words[k] = typo(words[k])
    if rng.random() < 0.15:
        words = words[1:] + words[:1]                    # word order
    a = addr
    if rng.random() < 0.4:
        a = " ".join(ABBR.get(w, w) for w in a.split())
    if rng.random() < 0.2:
        a = a.rsplit(",", 1)[0]                          # city dropped
    if rng.random() < 0.1:
        a = ""                                           # address missing
    return " ".join(words), a


def make_data():
    s1, cand, truth = [], [], defaultdict(set)
    for n in range(N_BUSINESSES):
        c = rng.choice(["US", "India"])
        name = f"{rng.choice(FIRST)} {rng.choice(TRADE)} {rng.choice(LEGAL[c])}".strip()
        addr = f"{rng.randint(1, 400)} {rng.choice(STREET[c])}, {rng.choice(CITY[c])}"
        sid = f"S1-{n:05d}"
        s1.append((sid, name, addr, c))
        n_copies = 0 if rng.random() < 0.06 else rng.choice([1, 2, 3, 4, 5])   # ~6 % singletons
        for _ in range(n_copies):
            nm, ad = noisy_copy(name, addr)
            cid = f"{rng.choice(['S2', 'S3'])}-{len(cand):06d}"
            cand.append((cid, nm, ad, c))
            truth[sid].add(cid)
        if rng.random() < 0.15:  # a look-alike DIFFERENT business next door: the trap that hurts precision
            w = name.split()
            w[1] = rng.choice(GENERIC)
            num = int(addr.split()[0]) + rng.randint(0, 3)
            cand.append((f"S3-{len(cand):06d}", " ".join(w), f"{num} {addr.split(' ', 1)[1]}", c))
    for sid, *_ in s1:
        truth.setdefault(sid, set())
    return s1, cand, truth


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalisation (real version: src/prep.py, ~300 lines of rules)
# ─────────────────────────────────────────────────────────────────────────────
EXPAND = {v.lower(): k.lower() for k, v in ABBR.items()}
LEGAL_WORDS = {"llc", "inc", "corp", "corporation", "pvt", "private", "ltd", "limited", "llp"}


def norm(s):
    toks = re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
    return " ".join(EXPAND.get(t, t) for t in toks)


def core(name):
    return " ".join(t for t in norm(name).split() if t not in LEGAL_WORDS)


def skeleton(tok):
    """Consonant skeleton: 'traders' and 'tradres' both become 'trdrs'. Robust to vowel typos."""
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouy]", "", tok))


def house_number(addr):
    m = re.match(r"\s*(\d+)", addr)
    return m.group(1) if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. Blocking: cheap keys -> a few candidates per Source 1 record
# ─────────────────────────────────────────────────────────────────────────────
def keys(name, addr, country):
    sk = sorted({skeleton(t) for t in core(name).split() if len(t) > 1})
    out = {f"tok:{t}" for t in sk}
    out |= {f"pair:{a} {b}" for i, a in enumerate(sk) for b in sk[i + 1:]}
    hn = house_number(addr)
    if hn:
        out |= {f"num:{t} {hn}" for t in sk}
    return {f"{country}|{k}" for k in out}             # country-scoped: never compare across countries


def block(s1, cand):
    index = defaultdict(list)
    for j, (_, nm, ad, c) in enumerate(cand):
        for k in keys(nm, ad, c):
            index[k].append(j)
    n = len(cand)
    idf = {k: math.log(n / len(v)) for k, v in index.items()}   # rare key = strong evidence
    pairs = []
    for i, (_, nm, ad, c) in enumerate(s1):
        score = Counter()
        for k in keys(nm, ad, c):
            for j in index.get(k, ()):
                score[j] += idf[k]
        for rank, (j, sc) in enumerate(score.most_common(K)):
            pairs.append((i, j, sc, rank))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Features for each candidate pair
# ─────────────────────────────────────────────────────────────────────────────
def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a and b else -1.0


def featurise(pairs, s1, cand):
    name_count = Counter(core(r[1]) for r in s1)       # rarity: how many S1 share this exact core name
    X = []
    for i, j, bscore, rank in pairs:
        n1, a1, n2, a2 = core(s1[i][1]), norm(s1[i][2]), core(cand[j][1]), norm(cand[j][2])
        h1, h2 = house_number(a1), house_number(a2)
        X.append([
            bscore, rank,
            SequenceMatcher(None, n1, n2).ratio(),        # character similarity of names
            jaccard(n1, n2),                              # shared words
            SequenceMatcher(None, a1, a2).ratio() if a1 and a2 else -1,
            -1 if not h1 or not h2 else int(h1 == h2),    # house number agrees?
            int(a2 == ""),                                # candidate has no address
            name_count[n1],                               # name rarity
        ])
    return np.array(X, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5-6. Decision layer and the metric
# ─────────────────────────────────────────────────────────────────────────────
def decide(pairs, prob, t):
    """One-to-one: each candidate record goes only to its most probable S1, and only if prob >= t."""
    best = {}
    for (i, j, *_), p in zip(pairs, prob):
        if p >= t and (j not in best or p > best[j][1]):
            best[j] = (i, p)
    out = defaultdict(set)
    for j, (i, _) in best.items():
        out[i].add(j)
    return out


def macro_f05(pred, truth_idx, s1_ids):
    total = 0.0
    for i in s1_ids:
        p, t = pred.get(i, set()), truth_idx.get(i, set())
        if not p and not t:
            total += 1
            continue
        tp = len(p & t)
        if tp:
            P, R = tp / len(p), tp / len(t)
            total += 1.25 * P * R / (0.25 * P + R)
    return total / len(s1_ids)


def main():
    s1, cand, truth = make_data()
    n_true = sum(len(v) for v in truth.values())
    print(f"[data]     {len(s1)} Source 1, {len(cand)} Source 2/3 records, {n_true} true links, "
          f"{sum(not v for v in truth.values())} singletons")

    cid = {c[0]: j for j, c in enumerate(cand)}
    truth_idx = {i: {cid[c] for c in truth[s[0]]} for i, s in enumerate(s1)}

    pairs = block(s1, cand)
    y = np.array([int(j in truth_idx[i]) for i, j, *_ in pairs])
    found = int(y.sum())
    print(f"[blocking] {len(pairs)} candidate pairs ({len(pairs) / len(s1):.1f} per S1) instead of "
          f"{len(s1) * len(cand):,} all-pairs; candidate recall {found / n_true:.1%} (the ceiling for everything after)")

    X = featurise(pairs, s1, cand)
    groups = np.array([i for i, *_ in pairs])
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, groups):   # all pairs of one S1 stay in the same fold
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
        oof[va] = model.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]

    ids = range(len(s1))
    naive = macro_f05(decide(pairs, X[:, 2], 0.8), truth_idx, ids)   # rule: name similarity >= 0.8
    print(f"[baseline] rule 'name similarity >= 0.8'            -> macro F0.5 {naive:.4f}"
          "   (many businesses share a name!)")
    rule2 = np.where((X[:, 2] >= 0.8) & (X[:, 5] == 1), 1.0, 0.0)
    smarter = macro_f05(decide(pairs, rule2, 0.5), truth_idx, ids)
    print(f"[baseline] rule 'name >= 0.8 AND same house number' -> macro F0.5 {smarter:.4f}"
          "   (misses copies with no address)")
    results = [(macro_f05(decide(pairs, oof, t), truth_idx, ids), t) for t in np.arange(0.2, 0.96, 0.05)]
    for score, t in results[::3]:
        print(f"[model]    threshold {t:.2f}                               -> macro F0.5 {score:.4f}")
    best = max(results)
    print(f"[model]    best threshold {best[1]:.2f} (chosen on out-of-fold predictions) -> macro F0.5 {best[0]:.4f}")


if __name__ == "__main__":
    main()
