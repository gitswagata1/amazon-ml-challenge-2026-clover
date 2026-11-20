# Learning track and exercises

A hands-on path from "never trained a model" to "could have contributed to this solution". Each level builds on the last.
Levels 1–3 need only a laptop. Answers to the "predict first" questions are at the bottom.

## Level 1 · The metric (30 min)

File: `examples/f05_metric.py`

1. Run it and check every printed number by hand with the formula.
2. **Predict first:** a business has 2 true copies. Option A: predict both plus 1 wrong. Option B: predict only 1 correct.
   Which scores higher?
3. Change `beta2 = 0.5 ** 2` to `1.0` (plain F1). Re-run. Which examples change the most, and why?
4. Explain in one sentence why the team used a high threshold.

## Level 2 · The toy pipeline (2 h)

File: `examples/toy_entity_resolution.py`

1. Run it. Write down the blocking recall and the final F0.5.
2. **The recall ceiling.** Set `K = 2`, then `K = 30`. Record recall and F0.5 each time. Plot F0.5 against recall. What
   is the relationship?
3. **Blocking keys.** In `keys()`, remove the `num:` keys. What happens to recall? Now remove `pair:` keys instead.
4. **Feature importance by ablation.** In `featurise()`, delete the `name_count` (rarity) feature. How much does F0.5 drop?
   Then delete the house-number feature instead. Which matters more here, and why?
5. **Leakage on purpose.** Replace `GroupKFold(5)` with `KFold(5, shuffle=True)`. Does the CV score go up or down?
   Why is the higher number *less* trustworthy?
6. **One-to-one.** In `decide()`, remove the "best S1 only" logic (accept every pair above `t`). What changes?
7. **Look-alikes.** Raise the 0.15 look-alike rate in `make_data()` to 0.5. Which threshold is best now? Compare with
   France's 0.91 vs US/India's 0.58.

## Level 3 · Read the real code (half a day)

1. Run `python code/code/business_entity_resolution/src/prep.py` (needs `pip install polars`). Read every assertion and
   write a one-line note on what noise it guards against.
2. Compute by hand: `skel("enterprises")`, `skel("enterpires")`, `skel("marketing")`.
3. Read `run.decide` (5 lines) and `run.f05`. Map them onto your toy pipeline.
4. Read `blocking._rescore`. Why is a shared house number worth +40, and why is a missing address given 50?
5. Read [the journey](journey.md)'s bug table. For each bug, decide which line of code fixed it.

## Level 4 · Models (1–2 days, a free Kaggle GPU helps)

1. **Embeddings.** Load `intfloat/multilingual-e5-small` with `transformers`. Embed `"query: Sharma Traders | MG Road"` and
   the Hindi-script version (`"query: शर्मा ट्रेडर्स | MG Road"`) and compute their cosine similarity. Try a look-alike
   (`Sharma Textiles`). Does the model rank them correctly?
2. **Cross-encoder.** Take the toy dataset's candidate pairs, fine-tune e5-small as a cross-encoder for 1 epoch (copy the
   training loop from `ce.py`), and add its score as a feature. How much does F0.5 improve?
3. **Stacking.** Re-create `rerank.py stack` on the toy data: band = OOF probability in [0.02, 0.98], meta-features
   `[logit(p), ce]`, compared with a control `[logit(p)]`.
4. **Unseen domain.** Make the toy generator produce a third "country" with different street words and more look-alikes.
   Train on the first two only. Then try P5-style self-training: confident positives + synthetic one-word-swap negatives.
   Does it help?

## Level 5 · Think like a competitor

1. Write a one-page plan for the 2027 challenge using [the lessons](lessons.md)'s checklist, *before* you see the data.
2. Pick one idea from [TECHNICAL §8](../TECHNICAL.md#8-gap-analysis-why-we-did-not-rank-higher)'s table and design the experiment that would test it: data, metric, noise bar, and what
   result would make you adopt it.
3. Read `research/SUBMISSIONS.md` top to bottom. For three rejected ideas, explain in your own words *why* they failed.

---

### Answers to "predict first"

- **Level 1, Q2:** Option A: P = 2/3, R = 1 → F0.5 = 0.714. Option B: P = 1, R = 1/2 → F0.5 = 0.833. **B wins.** Missing
  half the copies beats one wrong extra.
- **Level 2, Q5:** CV usually goes **up** with plain KFold, because pairs of the same business sit in both training and
  validation. The model partly memorises businesses, so it looks better than it will be on new businesses. The higher
  number is optimistic, not better.
- **Level 3, Q2:** `enterprises` → remove vowels, `y` and `h` → `ntrprss` → merge the repeated `s` → **`ntrprs`**; `enterpires` → **`ntrprs`** (same skeleton, so the typo is absorbed); `marketing` → **`mrktng`**.
  Vowels and `y`/`h` are removed, then repeated consonants are merged.

---

<sub>← [Technical deep dive](../TECHNICAL.md) · [Home](../README.md) · [The journey](journey.md) →</sub>
