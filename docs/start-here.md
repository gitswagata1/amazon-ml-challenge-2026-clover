# Start here

This repository is a complete, real machine-learning project, documented so that someone who has never trained a model
can follow it from the first line of data to the final submission, and so that someone experienced can audit every
decision.

## Who this is for

| You are… | Read this path |
|---|---|
| **New to ML** (you can read a bit of Python) | [problem](problem.md) → [build it in 6 steps](build-guide.md) → [TECHNICAL §0–2](../TECHNICAL.md) → [learning track](learning-track.md) · [glossary](../TECHNICAL.md#10-glossary) any time |
| **A teammate who wasn't in the code** | [problem](problem.md) → [journey](journey.md) → [TECHNICAL §0–2, §7](../TECHNICAL.md) → [debrief script](debrief-script.md) |
| **An ML engineer / reviewer** | [README](../README.md) → [TECHNICAL](../TECHNICAL.md) (all of it) → [code walkthrough](code-walkthrough.md) |
| **Presenting this to a group** | [debrief script](debrief-script.md), then diagrams from [TECHNICAL](../TECHNICAL.md) and the [journey](journey.md) |

## What you need to know before starting

Nothing beyond basic Python reading. Every term is explained when it first appears. Four ideas carry the whole project,
and it helps to hold them in your head from the start:

1. **A model is a function that learns from examples.** You show it thousands of pairs of records labelled "same
   business" or "different business", and it learns which clues (similar names, same house number…) predict "same".
2. **You can't compare everything with everything.** 1.7 million × 10 million is 17 trillion comparisons. So first you
   shortlist a few plausible candidates cheaply (*blocking*), then judge only those carefully (*matching*).
3. **You must measure honestly.** A model looks great on data it has already seen. All the care in this project around
   "folds" and "out-of-fold" is about testing the model only on data it has never seen.
4. **The score decides the strategy.** The score here punishes wrong merges twice as hard as missed matches, so the
   whole system is tuned to say "match" only when it is very sure.

## Hands-on first (recommended)

Reading about ML is like reading about swimming. Run these two files before [TECHNICAL §2](../TECHNICAL.md#2-the-pipeline-stage-by-stage):

```bash
pip install numpy scikit-learn
python examples/f05_metric.py               # the scoring rule, with worked examples
python examples/toy_entity_resolution.py    # the whole pipeline in miniature (~5 seconds)
```

Expected output of the toy pipeline:

```text
[data]     3000 Source 1, 8876 Source 2/3 records, 8422 true links, 169 singletons
[blocking] 30000 candidate pairs (10.0 per S1) instead of 26,628,000 all-pairs; candidate recall 92.7% (...)
[baseline] rule 'name similarity >= 0.8'            -> macro F0.5 0.3980   (many businesses share a name!)
[baseline] rule 'name >= 0.8 AND same house number' -> macro F0.5 0.8587   (misses copies with no address)
...
[model]    best threshold 0.80 (chosen on out-of-fold predictions) -> macro F0.5 0.9361
```

Three lessons are already visible:
- Blocking shrank the work 900× and still kept 92.7 % of true links.
- A name-only rule fails badly, because many businesses share a name. Adding the address helps a lot.
- A learned model that combines many clues beats both hand-written rules.

The real pipeline is the same idea with far better text cleaning, far more candidates found, ~53 clues instead of 8,
and transformers on top.

## How the docs are written

- **Plain first, technical second.** Each section explains the idea in everyday words, then gives the exact technical
  detail (file, function, number).
- **Links to the code** point to the exact commit of the competition code
  ([`95b58b6`](https://github.com/Martial-Gautam/Amazon-ML-Challenge-2026/tree/95b58b6f460fe362b6a4d2d362d3fb9845a1017c)),
  so they never go stale.
- **Facts vs opinion.** Every score and setting comes from the code or the team's documentation. Where we give an
  opinion (mostly [TECHNICAL §8](../TECHNICAL.md#8-gap-analysis-why-we-did-not-rank-higher)), it is labelled.
- **`[not in the files]`** marks anything we could not verify. [the open questions](open-questions.md) collects these.

---

<sub>← [Home (README)](../README.md) · [Home](../README.md) · [The problem](problem.md) →</sub>
