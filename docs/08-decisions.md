# 08 · Key decisions and trade-offs

These are the eight decisions that shaped the result. For each: what was decided, the alternative, the evidence, and the
price paid.

## 1. Frame it as pairwise classification + one-to-one assignment, not clustering

- **Chosen:** score (S1, candidate) pairs, then give each S2/S3 record to at most one S1.
- **Alternative:** cluster all records from all sources together (graph clustering, transitive closure).
- **Evidence:** Source 1 is deduplicated, so S1 records are natural cluster centres. Transitive S2–S3 chaining was tried
  and gave no gain: chains spread false merges, and F0.5 punishes those. The one-to-one rule won in every threshold search.
- **Price:** no use of S2↔S3 evidence (two candidates agreeing with each other).

## 2. Spend the effort on candidate recall, under a size budget

- **Chosen:** keep adding candidate sources (keys → rescue → embeddings → retriever → rare-empty) until recall hit 99.2 %,
  but keep only the top 20 from keys and accept a source only if it pays for its extra pairs.
- **Alternative:** a bigger k (30+), or a fancier matcher.
- **Evidence:** 56 % of v8's error was never-found matches. k = 30 vs 20 gave the same CV with 33 % more pairs.
- **Price:** ~25 candidates per business. That is small, but the organisers rewarded smaller sets, and top 10 (0.9468)
  would have cost too much F0.5.

## 3. LightGBM as the main matcher, transformers only where they pay

- **Chosen:** gradient-boosted trees over engineered features for all 43.9 M pairs; e5-base cross-encoders only on the
  ~1.5 M uncertain pairs.
- **Alternative:** a cross-encoder on everything, or a bigger transformer.
- **Evidence:** e5-base as a full feature was worth only +0.0005, and e5-large added ~0 and hurt France. On the uncertain
  band, the stack gave +0.00074 CV plus France gains. The small e5 cross-encoder as a feature gave +0.0077.
- **Price:** two model families to maintain, and a fiddly out-of-fold scheme to avoid leakage.

## 4. Out-of-fold discipline everywhere

- **Chosen:** GroupKFold by S1; neural models trained on held-out businesses (`--exclude-s1`) or in two out-of-fold
  halves; assertions that fail loudly on leakage.
- **Alternative:** train everything on everything (simpler, and faster to set up).
- **Evidence:** US/India leaderboard 0.9897 vs CV 0.98921. CV was trustworthy, which let the team make dozens of
  decisions without spending uploads.
- **Price:** ~100k training businesses kept away from the main model, and every neural model trained twice.

## 5. A noise bar for accepting changes

- **Chosen:** keep a change only if CV improves by more than ~+0.0005 (features +0.001; unseen-country work +0.002–0.003).
- **Alternative:** keep anything positive.
- **Evidence:** many "+0.0002" results were later shown to be noise (the second Optuna round, v8.1's word TF-IDF, which
  then hurt France on the leaderboard).
- **Price:** a few real but tiny gains were probably thrown away. That is a good trade under F0.5 and a deadline.

## 6. Use unlabelled French test text, with the organisers' permission

- **Chosen:** self-train French cross-encoders on pairs built from French test records (no labels), after asking the
  organisers.
- **Alternative:** treat France exactly like US/India and hope it transfers.
- **Evidence:** about +0.011 on France, the largest France gain. Fitting label-free statistics (TF-IDF, rarity) on
  train + test text was also needed, because a train-only vocabulary knows no French words.
- **Price:** some reliance on the public leaderboard to validate French decisions.

## 7. Two builds, merged by country

- **Chosen:** US/India rows from v10 (with retriever), French rows from v9 (without).
- **Alternative:** one consistent build.
- **Evidence:** hybrid 0.985926 vs v10 alone 0.98533 (+0.0006). The retriever's extra French candidates were mostly
  look-alikes.
- **Price:** more complexity. `reproduce.py` must run both builds, and the merge asserts that nothing outside France
  changed.

## 8. Precision-first thresholds, different per country

- **Chosen:** US/India 0.58 (CV-optimal with ambiguity rejection, flat from 0.56 to 0.62); France 0.91 (leaderboard sweep
  plus a fitted calibration curve).
- **Alternative:** one global threshold.
- **Evidence:** the French model is over-confident on look-alikes, and stricter French acceptance gained ~+0.0003 LB.
- **Price:** France's threshold was fitted to the public leaderboard, a known overfitting risk (chapter 07).

Next: [09 · Why we did not rank higher →](09-gap-analysis.md)
