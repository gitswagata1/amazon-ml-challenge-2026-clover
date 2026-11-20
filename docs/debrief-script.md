# 10-minute team debrief script

A spoken walkthrough for presenting the project to the group. The timings add up to about 10 minutes. Text in *italics*
is a suggestion for what to show on screen.

---

### 0:00 · Opening (30 s)

"I'll walk you through what we built for the Amazon ML Challenge, in plain words. We landed in the top 500 with a
public score of 0.9867. By the end, you should be able to explain to anyone what the problem was, how we solved it, and
where we lost points."

### 0:30 · The problem (1 min 30 s)

*Show the Sharma Traders table from [the problem guide](problem.md).*

"Amazon gave us business records from three sources. Source 1 is a clean list of about 1.7 million businesses. Sources 2
and 3 have about 10 million messy records of the same businesses: abbreviations, typos, Hindi or Tamil script, missing
addresses. There is no shared ID. For each Source 1 business, we had to list every matching record.

"Two twists. First, the score punishes a wrong match twice as hard as a missed one. Adding one wrong record hurts as much
as missing half the right ones, so we had to be cautious. Second, the test set had France, which never appears in
training. No French examples at all."

### 2:00 · The big idea: shortlist, then judge (1 min 30 s)

*Show the three-box funnel diagram from [TECHNICAL §1](../TECHNICAL.md#1-how-to-think-about-the-problem).*

"You can't compare 1.7 million businesses with 10 million records. That's 17 trillion comparisons. So every solution
does two things. First, a cheap shortlist: for each business, find about 25 records that might match. That's called
blocking. Then a careful judge decides which of those 25 really match.

"The key insight: if the shortlist misses a true match, the judge can never find it. So most of our big jumps came from
improving the shortlist, not the judge. We went from 81 % of true matches in the shortlist on day 1 to 99.2 % at the end,
and the score went from 0.85 to 0.98."

### 3:30 · How the shortlist works (1 min)

"First we clean the text. We convert Indian scripts to English letters, expand 'Rd' to 'road', and remove 'Pvt Ltd'.
We even built a small dictionary from the training data, because the data translated English words into Indian scripts.
For example, 'global' in Tamil script maps back to 'global'.

"Then each record gets keywords, things like 'name word plus house number'. Records sharing rare keywords become
candidates. On top of that, we used a multilingual AI model that finds records with similar *meaning*, which catches
matches that share no keyword at all."

### 4:30 · How the judge works (1 min 30 s)

*Show the architecture diagram in the README.*

"For every candidate pair we compute about 53 numbers: how similar the names are, whether house numbers match, and
importantly, how *rare* the name is. 'Sai Traders' with no address could be anyone. A rare name with no address is
probably the same business. That rarity feature was our biggest single feature gain.

"A model called LightGBM, which is many small decision trees, turns those numbers into a probability. For the pairs it's
unsure about, about 1.5 million of them, a transformer reads both records side by side like a human would, and a small
model combines both opinions. Finally, each record goes to at most one business, and we accept a match only above a
strict threshold."

### 6:00 · France (1 min 30 s)

"France was the hardest part. No labels, a new language, and far more look-alike businesses at the same address, like
'Chti Parents' and 'Chti Centre'. Two things worked.

"One: rules that copy how the dataset was generated. Changing one generic word usually means a *different* business.
Two: we taught the transformer French without any labels. We took French pairs the model was 99.5 % sure about as
examples of matches, and created fake look-alikes by swapping one word, as examples of non-matches. Fine-tuning on that
gained about 1 point on France. The organisers confirmed this was allowed."

### 7:30 · What went wrong (1 min)

"Honest version. France ended at about 0.967 while US and India were at 0.990. That gap is basically the distance between
us and the top 10. We also had a power cut, an out-of-memory crash that killed every job, GPU quotas running out, and
a Kaggle job that silently used an old dataset. Because every step saved its output, we could always resume. And we
tested and rejected more than 30 ideas, each with a written reason."

### 8:30 · Lessons (1 min)

"Three things to take away. One: find your ceiling first. Our analysis showed 56 % of our error came from matches the
shortlist never found. Two: validate exactly like the leaderboard scores you. Our own estimate for US/India was within
0.001 of the real score. Three: if part of the test is unseen, like France, plan for it from day one, not the last day."

### 9:30 · Close (30 s)

"Everything is documented in the repo: the full pipeline, every model, every failure, and a small toy version you can run
on your laptop in five seconds to see the whole thing work. Questions?"

---

**Likely questions and short answers**

| Question | Answer |
|---|---|
| Why not just use ChatGPT on every pair? | 44 million pairs; too slow and expensive, and the rules capped models at 8 B parameters with MIT/Apache licences. |
| Why LightGBM and not deep learning only? | Tabular features, tens of millions of rows, CPU training. Trees are the best tool there; transformers were used where they paid. |
| Wasn't using test data cheating? | We used only the text, never labels, and the organisers confirmed it was allowed. |
| What would you do differently? | Work on France from day 1: build fake French training data from US/India labels, and a French-aware shortlist. |

---

<sub>← [Code walkthrough](code-walkthrough.md) · [Home](../README.md) · [Open questions](open-questions.md) →</sub>
