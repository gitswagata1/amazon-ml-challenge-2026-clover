# 01 · The problem in plain English

## The one-sentence version

Amazon gave us three lists of businesses that overlap but use no shared ID, and asked: **for every business in the clean
list, which records in the other two lists are the same business?**

This task is called **entity resolution** (ER): deciding which records refer to the same real-world thing (the *entity*).

## Why Amazon cares

Picture a seller signing up on Amazon. Their business details arrive from a tax registry, a payment provider, a
logistics partner, and the seller's own form. Each source spells the name differently, abbreviates the address
differently, or is missing pieces:

| Source | Name | Address |
|---|---|---|
| 1 (reference) | Sharma Traders Private Limited | 12, MG Road, Indiranagar, Bengaluru 560038 |
| 2 | SHARMA TRADERS PVT LTD | 12 M.G. Rd, Bangalore |
| 3 | शर्मा ट्रेडर्स | Near SBI ATM, MG Road, 560038 |
| 3 | Sharma Trading Services | 14 MG Road, Bengaluru *(a different business next door!)* |

A human sees at once that the first three rows are the same shop and the fourth probably isn't. The challenge is to
make a machine do this reliably across **millions** of records. Getting it wrong has a cost:

- **A false merge** joins two different businesses. Payments, reviews or compliance flags from one leak onto the other.
  This is the dangerous mistake.
- **A missed match** leaves one business split in two. This is annoying but less harmful.

## What exactly we had to produce

- **Source 1** is *deduplicated*: every row is a different business.
- For **each** Source 1 business, list the IDs of every matching record from Source 2 and Source 3. The list can be
  empty (a *singleton*), have one entry, or have many.
- Records only carry four fields: `entity_id` (prefix `S1-`, `S2-`, `S3-`), `business_name`, `business_address`, `country`.
- Training data covers **US and India** and includes the answers (`train_ground_truth.tsv`).
- Test data covers US, India **and France**, which never appears in training. The rules said: treat country as an open
  set, never hard-code {US, India}.

Two files had to be submitted:

| File | What it holds | Scored? |
|---|---|---|
| `matching_results.tsv` | `source1_entity_id` → comma-separated matched IDs | **Yes** (leaderboard) |
| `candidate_pairs.tsv` | `source1_entity_id` → the candidate IDs the model actually scored | Reviewed for the final ranking: **smaller candidate sets rank higher** |

That second rule matters. Amazon resolves billions of records, so a solution that only works by comparing each business
with thousands of candidates is useless to them. The candidate set had to be small (ours averages ~25 per business).

## How we were scored: F0.5, explained slowly

For **each** Source 1 business, compare our predicted list with the true list:

- **Precision** = of the records *we* said match, what fraction really do? (Are we trustworthy?)
- **Recall** = of the records that *truly* match, what fraction did we find? (Are we thorough?)

**F-beta** combines the two into one number. With β = 0.5, precision counts **twice** as much as recall:

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Special cases:
- A true singleton (no matches) where we predict **nothing** scores **1.0**.
- A true singleton where we predict **anything** scores **0.0**.
- If none of our predictions are right, the score is 0.

The final score is the **macro average**: compute F0.5 per Source 1 business, then take the plain mean. A business with 1
copy counts exactly as much as one with 10.

### Worked examples

Run `python examples/f05_metric.py` to reproduce these.

| Situation (truth = 4 copies) | Precision | Recall | F0.5 |
|---|:---:|:---:|:---:|
| Found all 4 + 1 wrong extra | 0.80 | 1.00 | **0.833** |
| Found 3 of 4, nothing wrong | 1.00 | 0.75 | **0.938** |
| Found 2 of 4, nothing wrong | 1.00 | 0.50 | **0.833** |

Read that table twice. **Missing one copy (0.938) is much better than adding one wrong copy (0.833).** Adding one wrong
record costs as much as missing *half* the true ones. This one fact shaped the whole solution:
- We only say "match" when the model is quite sure (threshold 0.58 for US/India and 0.91 for France, on a 0–1 scale).
- We built a whole layer of rules to *reject* look-alike businesses.
- We assign each Source 2/3 record to **at most one** business, because Source 1 has no duplicates, so a record can't
  belong to two of them.

The official example: predict `[S2-00047, S2-00193, S3-00812]`, truth `[S2-00047, S3-00812]`. Precision = 2/3, recall = 1,
F0.5 = 0.714.

## The rules (constraints)

| Rule | What it meant for us |
|---|---|
| **No external data.** No business registries, geocoding APIs or internet lookups | Everything, including the Indian-script dictionary, had to be learned from the given data |
| **Models must be MIT or Apache 2.0 licensed and ≤ 8 B parameters** | We used `multilingual-e5-small` (118 M) and `-base` (278 M), both MIT, and LightGBM (MIT) |
| **72-hour challenge window** (25–27 Sept 2026) | Every experiment had to be cheap to re-run, hence heavy caching |
| Every test Source 1 record gets exactly one row, with no duplicates and only S2/S3 IDs | A validator script (`utils/validate_submission.py`) was run before every upload |
| Public leaderboard uses part of the test set; final ranking uses the private remainder | Tuning too hard against the public score is risky (see chapter 07) |
| Final package includes code that reproduces both files | Hence `reproduce.py` |

Next: [02 · The data →](02-data.md)
