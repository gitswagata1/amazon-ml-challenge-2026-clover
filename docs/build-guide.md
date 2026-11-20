# Build it from scratch in 6 steps

This guide rebuilds the core of the competition solution on a laptop, one step at a time. Paste each step's code into
**one** Python file, in order, and run it from the repo root after each step. The complete file is
[`examples/build_from_scratch.py`](../examples/build_from_scratch.py) if you get stuck.

```bash
pip install -r examples/requirements.txt     # numpy, scikit-learn
python examples/build_from_scratch.py          # ~10 seconds, all six steps
```

| Step | You build | The real pipeline's version |
|---|---|---|
| 1 | macro F0.5 | `run.f05`, `tune.macro_f05` |
| 2 | a dataset with realistic noise | 2.2 M + 10.3 M training records (`blocking.ensure`) |
| 3 | IDF-weighted key blocking | `prep.block_keys`, `blocking.block` + 4 more candidate sources |
| 4 | 7 pair features | ~53 features (`features.py`, `extra_features.py`, `embed.py`, `ce.py`) |
| 5 | gradient boosting + GroupKFold | LightGBM, Optuna, 3 seeds + cross-encoder stacking |
| 6 | one-to-one + threshold | `run.decide`, `decide.py`, French rules |

## Step 1 · The metric

**Goal.** Write the score before you write a model. You can't improve what you can't measure, and the metric decides the whole strategy.

**Code.**

```python
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
```

**Expected output.**

```text
step 1: 0.714 0.833
```

**Check your understanding.** Why is predicting `{a}` (one right, one missed) better than `{a, b, x}` (both right, one wrong) under F0.5?

<details><summary>Answer</summary>

F0.5 weights precision twice as much as recall. `{a, b, x}` has precision 2/3, which hurts more than `{a}`'s recall of 1/2. So when unsure, leave it out.

</details>

## Step 2 · The data

**Goal.** Look at the data before modelling. Find out how many records there are, how many links, how many singletons, and what a match actually looks like.

**Code.**

```python
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
```

**Expected output.**

```text
step 2: 3000 S1 | 8876 candidates | 8422 true links | 169 singletons
  S1  : ('Vision Consulting', '26 Oak Avenue, Austin')
  copy: ('Conuslting Vision', '')
  copy: ('Vision Consulting', '26 Oak Avenue, Austin')
```

**Check your understanding.** The second copy has an empty address and swapped, misspelled words. Which features would still catch it?

<details><summary>Answer</summary>

Name-based ones that ignore order and tolerate typos (token-set similarity, consonant skeletons). Address features must treat *empty* as *unknown*, not as *different*.

</details>

## Step 3 · Blocking (the shortlist)

**Goal.** Never compare everything with everything. Give each record a few cheap keys, weight rare keys higher (IDF), and keep the top K candidates per business. Measure **candidate recall**: it is the ceiling for everything after.

**Code.**

```python
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
```

**Expected output.**

```text
step 3: K= 2 ->   6000 pairs, candidate recall 57.4%
step 3: K= 5 ->  15000 pairs, candidate recall 89.1%
step 3: K=10 ->  30000 pairs, candidate recall 92.7%
step 3: K=20 ->  60000 pairs, candidate recall 97.7%
```

**Check your understanding.** Going from K = 10 to K = 20 doubles the pairs. Is it worth it?

<details><summary>Answer</summary>

It depends on the cost of pairs. In the competition, smaller candidate sets ranked higher, and the team found top 20 vs 30 gave the same score with 33 % fewer pairs. Measure recall *per pair added*.

</details>

## Step 4 · Features

**Goal.** Turn each candidate pair into numbers a model can learn from: name and address similarity, house-number agreement, missing-address flags, and name rarity.

**Code.**

```python
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
```

**Expected output.**

```text
step 4: (30000, 7) feature matrix | 26.0% of pairs are true matches
```

**Check your understanding.** Two records are both named "Prime Traders". Why does `name_count` matter?

<details><summary>Answer</summary>

If 12 businesses are called "Prime Traders", a name match alone proves little. If the name is unique, it is strong evidence. In the real pipeline, rarity was the biggest single feature gain (+0.0038).

</details>

## Step 5 · The model

**Goal.** Train a gradient-boosted tree model on the features, validated with **GroupKFold** so no business appears in both training and validation. Keep the **out-of-fold** probabilities: honest predictions for every pair.

**Code.**

```python
# ── Step 5 · the model ───────────────────────────────────────────────────
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

oof = np.zeros(len(y))                                    # out-of-fold probabilities
for tr, va in GroupKFold(5).split(X, y, groups):          # one business never in train AND validation
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
    oof[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
print(f"step 5: out-of-fold AUC {roc_auc_score(y, oof):.4f}")
```

**Expected output.**

```text
step 5: out-of-fold AUC 0.9961
```

**Check your understanding.** What happens to the AUC if you swap `GroupKFold` for a shuffled `KFold`, and why is that number less trustworthy?

<details><summary>Answer</summary>

It usually goes up, because the model has seen other pairs of the same business. It partly memorises businesses, so the score is optimistic for new ones.

</details>

## Step 6 · Decisions

**Goal.** Turn probabilities into answers: each candidate record goes to at most one business (its best), and only above a threshold chosen on out-of-fold predictions with the real metric.

**Code.**

```python
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
```

**Expected output.**

```text
step 6: threshold 0.3 -> macro F0.5 0.9161
step 6: threshold 0.5 -> macro F0.5 0.9332
step 6: threshold 0.7 -> macro F0.5 0.9360
step 6: threshold 0.9 -> macro F0.5 0.9358
step 6: hand-written rule (name >= 0.8 and same house no.) -> 0.9303
```

**Check your understanding.** On this toy data the hand-written rule is close to the model. Why did the real competition need the model (and transformers)?

<details><summary>Answer</summary>

The toy data has one kind of noise and simple look-alikes. Real data had nine scripts, deliberate decoys, empty addresses and an unseen country. No short rule covers all of that, but a model combining ~53 signals can. You can see the start of this even here: raise the look-alike rate in `make_data()` from 0.15 to 0.6 and the gap roughly doubles (model 0.937 vs rule 0.926, measured). Every new kind of noise needs a new rule, while the model just learns it.

</details>

## Where to go next

- Change one thing at a time and re-run: K, the keys, a feature, the threshold. Write down what moved and why.
- Then do the exercises in [the learning track](learning-track.md) and read [TECHNICAL §2](../TECHNICAL.md#2-the-pipeline-stage-by-stage)
  to see how each of these six steps was scaled up for 20 million real records.

---

<sub>← [The data](data.md) · [Home](../README.md) · [Technical deep dive](../TECHNICAL.md) →</sub>
