"""Build entity resolution from scratch in 6 steps: the solution file for docs/build-guide.md.

Run from the repo root:  python examples/build_from_scratch.py   (needs numpy, scikit-learn; ~10 s)
"""
# ── Step 1 · the metric ──────────────────────────────────────────────────
def f05(pred: set, true: set) -> float:
    if not pred and not true:      # correctly predicted "no matches"
        return 1.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)

def macro_f05(pred: dict, truth: dict) -> float:
    return sum(f05(pred.get(k, set()), v) for k, v in truth.items()) / len(truth)

print("step 1:", round(f05({"a", "b", "x"}, {"a", "b"}), 3), round(f05({"a"}, {"a", "b"}), 3))

# ── Step 2 · the data ────────────────────────────────────────────────────
import sys; sys.path.insert(0, "examples")
from toy_entity_resolution import make_data

s1, cand, truth = make_data()          # s1/cand rows: (id, name, address, country)
cid = {c[0]: j for j, c in enumerate(cand)}
truth_idx = {i: {cid[c] for c in truth[r[0]]} for i, r in enumerate(s1)}   # S1 row -> set of cand rows
print("step 2:", len(s1), "S1 |", len(cand), "candidates |",
      sum(len(v) for v in truth_idx.values()), "true links |",
      sum(not v for v in truth_idx.values()), "singletons")
i = next(i for i, v in truth_idx.items() if len(v) >= 2)
print("  S1  :", s1[i][1:3])
for j in sorted(truth_idx[i]):
    print("  copy:", cand[j][1:3])

# ── Step 3 · blocking ────────────────────────────────────────────────────
import math, re
from collections import Counter, defaultdict

LEGAL = {"llc", "inc", "corp", "pvt", "private", "ltd", "limited", "llp"}
def tokens(name):
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", name.lower()).split() if t not in LEGAL and len(t) > 1}
def skel(t):                                   # drop vowels, merge repeats: typo-tolerant
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouy]", "", t))
def keys(name, addr, country):
    sk = sorted({skel(t) for t in tokens(name)})
    num = (re.match(r"\s*(\d+)", addr) or [None, ""])[1]
    ks = {f"t:{t}" for t in sk} | {f"p:{a} {b}" for i, a in enumerate(sk) for b in sk[i + 1:]}
    ks |= {f"n:{t} {num}" for t in sk} if num else set()
    return {country + "|" + k for k in ks}

def block(K):
    index = defaultdict(list)
    for j, (_, nm, ad, c) in enumerate(cand):
        for k in keys(nm, ad, c):
            index[k].append(j)
    idf = {k: math.log(len(cand) / len(v)) for k, v in index.items()}
    pairs = []
    for i, (_, nm, ad, c) in enumerate(s1):
        score = Counter()
        for k in keys(nm, ad, c):
            for j in index.get(k, ()):
                score[j] += idf[k]
        pairs += [(i, j, sc, rank) for rank, (j, sc) in enumerate(score.most_common(K))]
    return pairs

def recall(pairs):
    found = sum(j in truth_idx[i] for i, j, *_ in pairs)
    return found / sum(len(v) for v in truth_idx.values())

for K in (2, 5, 10, 20):
    print(f"step 3: K={K:2d} -> {len(block(K)):6d} pairs, candidate recall {recall(block(K)):.1%}")
pairs = block(10)

# ── Step 4 · features ────────────────────────────────────────────────────
import numpy as np
from difflib import SequenceMatcher

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio() if a and b else -1.0
def house(addr):
    m = re.match(r"\s*(\d+)", addr); return m.group(1) if m else ""

name_count = Counter(" ".join(sorted(tokens(r[1]))) for r in s1)     # how common is this name?
def features(i, j, bscore, rank):
    n1, n2 = " ".join(sorted(tokens(s1[i][1]))), " ".join(sorted(tokens(cand[j][1])))
    a1, a2 = s1[i][2].lower(), cand[j][2].lower()
    h1, h2 = house(a1), house(a2)
    return [bscore, rank, sim(n1, n2), sim(a1, a2),
            -1 if not h1 or not h2 else int(h1 == h2),   # unknown is not the same as "different"
            int(a2 == ""), name_count[n1]]

X = np.array([features(*p) for p in pairs], dtype=np.float32)
y = np.array([int(j in truth_idx[i]) for i, j, *_ in pairs])
groups = np.array([i for i, *_ in pairs])
print("step 4:", X.shape, "feature matrix |", f"{y.mean():.1%} of pairs are true matches")

# ── Step 5 · the model ───────────────────────────────────────────────────
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

oof = np.zeros(len(y))                                    # out-of-fold probabilities
for tr, va in GroupKFold(5).split(X, y, groups):          # one business never in train AND validation
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
    oof[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
print(f"step 5: out-of-fold AUC {roc_auc_score(y, oof):.4f}")

# ── Step 6 · decisions ───────────────────────────────────────────────────
def decide(prob, t):
    best = {}                                             # each candidate record -> its single best S1
    for (i, j, *_), p in zip(pairs, prob):
        if p >= t and (j not in best or p > best[j][1]):
            best[j] = (i, p)
    out = defaultdict(set)
    for j, (i, _) in best.items():
        out[i].add(j)
    return out

for t in (0.3, 0.5, 0.7, 0.9):
    print(f"step 6: threshold {t:.1f} -> macro F0.5 {macro_f05(decide(oof, t), truth_idx):.4f}")
rule = np.where((X[:, 2] >= 0.8) & (X[:, 4] == 1), 1.0, 0.0)
print(f"step 6: hand-written rule (name >= 0.8 and same house no.) -> {macro_f05(decide(rule, 0.5), truth_idx):.4f}")

