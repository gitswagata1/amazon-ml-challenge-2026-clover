# 06 · France: learning a country with no labels

France was 15 % of the test set, had **zero** training examples, and in the end accounted for almost all of our remaining
error:

| Part of the test set | Public LB score (diagnostic upload) |
|---|---|
| US + India | **0.9897** (slightly *above* our CV of 0.98921) |
| France | **0.9673** |

*How we know:* the team uploaded the final file with every French row emptied. Since French singletons were assumed to be
5.58 %, as in training, the change in score let them back out the two parts separately (submission #26). This is a neat
trick for measuring a subgroup when the leaderboard gives only one number.

## Why France is hard

1. **No labels.** No French example of "these two are the same" or "these two are different".
2. **New language, new conventions.** *Rue*, *Allée*, *Cedex*, *SARL*, *Ets*, *St-Nazaire* = Saint, `59bis`, elisions like `L'Atelier`.
3. **Far more traps.** France has **4× more uncertain pairs per business, 10–20× more ownership conflicts, and 10–40× more
   co-located look-alike businesses** than US/India. French names are often *city + generic word* ("Chti Parents" vs
   "Chti Centre" at the same address).
4. **The leaderboard is the only judge**, and uploads were limited.

In machine-learning terms this is **unsupervised domain adaptation**: you have labels in the *source domain* (US/India)
and must perform well in a *target domain* (France) where you only have unlabelled inputs.

## What transfers and what doesn't

The team measured transfer by pretending one country was unseen (*leave-one-country-out*):

| Setup | Train US → score India | Train India → score US |
|---|:---:|:---:|
| early LightGBM | 0.822 | 0.946 |
| + cross-encoder feature | 0.937 | 0.969 |

**Lesson:** text models that read words (cross-encoders) transfer across countries. Statistical features that depend on
a country's typical scores transfer less well.

Things that **failed** to transfer:
- **Importance weighting** (re-weighting training rows to look like the target country): a domain classifier separated
  countries almost perfectly (98 % of rows clipped), so the weights carried no signal. −0.0057 / +0.0016.
- **A learned corrector** trained on one proxy direction: negative on the other. It learned the source country's quirks.
- **Plain pseudo-labelling** of the cross-encoder: −0.036 in the US→India simulation.

Things that **did** transfer: rules derived from **how the data generator works**. These are the same in every country.

## Reverse-engineering the generator (from training labels)

The team studied US/India training labels to learn how noise copies and decoys were made:

- **Noise copies** (true matches): the last word is replaced by a word from a small list (US/India: *center*,
  *services*, *service*, *partners*). Such swaps were **97 % true matches**. Any other single-word swap was ≈ 0 % true.
- **Decoys** (different businesses): the S1 name + one extra word from a list (France: *france, groupe, developpement,
  holding, participations, international, distribution*), placed at a **nearby house number**. The model already
  rejected these (p ≈ 0).
- **The open group** (call it G): same address, *one non-typo word swapped*, no twin record, no S1 with that name.
  France had 113,605 such pairs. In training these were true matches **80 % of the time in the US but 38 % in India**, so
  France was unknown. The break-even precision for F0.5 here is about 0.72.

## The French toolkit, in the order it was built

### 1. Look-alike rejection rules (`france.py rules` / `apply`)

**P1, same-address swap.** Reject an accepted French pair if the names differ by exactly **one word removed and one
added** (at least one shared word), the swap isn't a typo (similarity < 70), both words are common (each in ≥ 20 French
S1 names), the candidate has a house number, no other record or S1 carries that name there, and the addresses are ≥ 95 %
similar.

- Tested with one leaderboard probe that removed 17,826 accepted French pairs: **+0.001118 LB → France ≈ +0.0075**. About
  70 % of the removed pairs had been wrong.

**P2′, different-address swap.** The same one-word swap at a real, different address (similarity < 95), restricted to
symmetric words. Screened in an unseen-country simulation before use. Four other rule variants (P3–P6) were rejected in
that screening.

### 2. Self-training the cross-encoder on French text (P5): the big win

`france.py self-train-data` builds a French training set **using only the French test records' text**. No labels are
used, and the organisers confirmed this is allowed:

| Part | How it is made | Count |
|---|---|---|
| Pseudo-positives | pairs the pipeline scored **≥ 0.995** that are their record's best match, **excluding** any pair flagged by the look-alike rules | 300,000 |
| Sibling negatives | take a positive's candidate name and **swap one generic French word** for another (40 % also shift the house number by 1–15) | up to 300,000 |
| Decoy negatives | candidate name + a decoy word (*Groupe, Holding, Développement…*) at a shifted house number | up to 100,000 |
| Hard negatives | different-address look-alikes (P2′ list) | 150,000 |
| Easy negatives | pairs the pipeline scored < 0.01 | 100,000 |
| **Total** | after skipping names that couldn't be modified | **885,573** |

"Generic French words" are mined automatically: frequent in French S1 names (≥ 200), not city names (frequent in
addresses), and not legal/noise words.

Then the `ce2` half-0 re-ranker is fine-tuned on this for 1 epoch (learning rate 2e-5, batch 64). It re-scores every
French pair with stage-1 probability ≥ 0.02, and the stacking model uses that score in place of `ce2`.

- Leaderboard probe, changing French decisions only: **+0.001688 LB → France ≈ +0.011**. This was the largest single
  France gain.
- **P5b**, the same recipe starting from the half-1 model, was averaged with P5: +0.000124 LB.

**Why did this work when plain pseudo-labelling failed?** Plain pseudo-labelling shows the model only its own easy,
confident decisions, so it learns nothing new and drifts toward its own biases. P5 adds **synthetic hard negatives
that encode the generator's trap logic in French words**. That is genuinely new information: "in France, swapping
*Parents* for *Centre* means a different business." It also gets French vocabulary from the positives. Keeping
rule-flagged pairs out of the positives stops it from learning the look-alike mistakes.

**Round 2 hurt.** Repeating the process with the improved model's labels cost −0.000184. Round-2 positives include the
model's confident look-alike mistakes, which the P1 probe showed are often wrong.

### 3. The France threshold (0.91), tuned on the leaderboard

With no French labels, CV can't pick a French threshold. The team swept it on the public leaderboard:

| France threshold | 0.76 | 0.80 | 0.88 | **0.91** | 0.92 | 0.93 | 0.96 |
|---|---|---|---|---|---|---|---|
| Public LB | 0.986128 | 0.986206 | 0.986335 | **0.986371** | 0.986368 | 0.986345 | 0.986191 |

They fitted a calibration curve to these results (`q = sigmoid(−0.90 + 0.77·logit p)`, residuals ≤ 9e-6). It predicted
0.986367 at 0.91, against an actual 0.986371. The peak is flat between 0.88 and 0.93, which reduces (but doesn't remove)
the risk of overfitting the public split.

Why so much higher than US/India's 0.58? French probabilities are over-confident, because the model has never been
corrected on French look-alikes. Being stricter compensates.

### 4. Recovery (`france.py recover`)

The rules sometimes removed true matches. Two reversals were added:
- Re-accept removed pairs whose swapped word is a **generator noise word**: *associés* (always) or *services* (if
  probability ≥ 0.5). **+0.000154 LB.**
- Re-accept removed pairs that are their record's best match with probability **≥ 0.91** (both French models confident).
  **+0.000148 LB**, then +0.000018 for the final remaining cases.

### 5. Which candidate build for France?

v10's fine-tuned retriever added French candidates that were mostly look-alikes. France from v9 (no retriever) scored
**+0.004 better** on France at the earlier threshold. At 0.92 the gap was only 0.000025, but v9 was kept. Hence the
final hybrid: US/India rows from v10, French rows from v9.

## Where France still loses points

French **recall ≈ 0.90**: about one in ten French copies is never accepted. In the last hours the team checked and ruled
out the obvious causes:
- Pairs with stage-1 probability < 0.02 that P5 might rescue: 0.25 % had a positive logit, and none were confident.
- Exact-name same-address copies missed by blocking: 38 (negligible).
- Twins, S1 siblings, acronyms: ≤ 500 pairs each.
- Removing low-confidence saturated French pairs: 99 % of them were true, so rejected.

Their conclusion: the missing French copies carry noise that neither blocking, LightGBM nor P5 recognises. Fixing it
would need a new French-aware matcher, which wasn't feasible before the deadline. Chapter 09 discusses what that could
have looked like.

Next: [07 · Bottlenecks and failures →](07-bottlenecks.md)
