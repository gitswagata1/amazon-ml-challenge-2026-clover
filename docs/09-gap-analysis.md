# 09 · Why we did not rank higher

> ⚠️ **This chapter is informed opinion, not fact.** We don't have the top teams' code or write-ups. The *facts* used
> here (our scores, the error breakdown, the leaders being above 0.99) come from the team's logs. The *explanations* and
> *suggestions* are our best reading of them.

## The facts

- Our final public score: **0.986691**. During the challenge, the team noted the **top-10 teams were above 0.99**.
- Our US/India rows scored **0.9897** on the leaderboard, roughly at the leaders' level.
- Our French rows scored **0.9673**. France is 15 % of the test set.

## The arithmetic of the gap

If France had scored as well as US/India:

```
0.85 × 0.9897 + 0.15 × 0.9897 ≈ 0.9897     (vs our 0.9867)
```

That is about **+0.003**, which is roughly the whole distance to the top 10. **Almost all of the gap is France.** A team
that solved France as well as it solved US/India would have been near the top with an otherwise similar pipeline.

## Why France stayed behind (opinion)

1. **France got serious attention late.** The French self-training (P5) was built on the final day, after v8–v10 work on
   candidates. It was the single biggest French lever (+0.011). A second, better-designed iteration (more synthetic
   noise types, rather than more pseudo-labels) might have added a similar amount again.
2. **French recall ≈ 0.90.** The remaining misses are "noisy variants neither blocking, stage 1 nor P5 recognises".
   They needed French-specific *candidate generation*, not just better French judging. The retriever was trained only on
   US/India, so it found French look-alikes rather than French copies.
3. **No French validation set.** Every French decision cost an upload. Leaders may have built a synthetic French
   validation set early, by applying French-style noise to labelled US/India pairs, and iterated on CV instead.

## Things top teams may have done that we didn't (opinion)

| Idea | Why it might have helped | Status here |
|---|---|---|
| **Generate French-style training data from labelled US/India pairs** (rewrite addresses into French formats, swap in French generic/legal words, apply generator-style noise) | It gives *labelled* French-like examples for every model, including the retriever, and a French validation set | not tried |
| **Train the retriever on France-like data** | The fine-tuned retriever helped US/India (+1.4 % recall) but hurt France; a French-aware retriever could have raised French recall | not tried |
| **An LLM judge on the hardest French pairs** — an Apache-2.0 open model ≤ 8 B (for example Qwen2.5-7B-Instruct or Mistral-7B, both Apache 2.0) | A general-knowledge model knows "Comité" ≠ "Club" and French naming conventions without labels | not tried (time and GPU budget) |
| **Earlier near-complete recall** | The team's own analysis said the leaders had near-complete recall while ours capped at 0.989 (v8). The retriever only arrived in v10, on the last day | done, but late |
| **More GPU** | ~30 GPU-hours/week per account limited how many neural variants could be trained | constraint |

## What this team did *better* than most (also opinion)

To keep the picture fair: the leaderboard-to-CV agreement on US/India (0.9897 vs 0.98921) shows unusually careful
validation. The documented rejection list, the per-country probes and the reproducible one-command pipeline are the
habits of a strong ML team, not a lucky one. Going from **~1,069th** after v7 to the **Top 500** within the same 72 hours
came from method, not from guessing.

## Unknowns

- The **private** leaderboard score and final rank: `[not in the files]`. Because several French decisions were tuned on
  the public leaderboard, the private score could be slightly lower.
- How much weight the organisers gave to candidate-set size in the final ranking: `[not in the files]`.

Next: [10 · Lessons for the next hackathon →](10-lessons.md)
