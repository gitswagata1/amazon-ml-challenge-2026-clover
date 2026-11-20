# Lessons for the next ML hackathon

A checklist distilled from what worked and what hurt. Most of it applies to any ML competition, not just entity resolution.

## Before writing a model

- [ ] **Read the metric until you can compute it by hand.** F0.5 decided the threshold, the one-to-one rule, the
      look-alike rules, and even which ideas were worth trying. Write the metric function first (`examples/f05_metric.py`).
- [ ] **Look for the "unseen" part of the test set.** Here it was France, 15 % of the test and almost all of the final gap.
      Plan for it on day 1, not day 3.
- [ ] **Do EDA that answers design questions.** "Do matches cross countries?" (no → block per country). "How are decoys
      made?" (one word + nearby number → rules and synthetic negatives). Each finding should change the plan.
- [ ] **Ask the organisers early** about grey areas (here: using unlabelled test text). The answer unlocked the biggest
      France gain.

## Validation

- [ ] **Validate with the competition metric**, grouped the way the test is grouped (GroupKFold by S1), including the
      cases your pipeline misses entirely.
- [ ] **Set a noise bar** and stick to it. Many +0.0002 "gains" were noise.
- [ ] **Re-check small-sample results at full size** before rejecting them (Optuna: +0.0002 quick, +0.0028 full).
- [ ] **Simulate unseen groups** (leave-one-country-out) when part of the test is new, but remember a simulation is only
      a proxy.
- [ ] **Use probe uploads that change one subgroup only**, so the score change belongs to that subgroup. Use an
      "empty the subgroup" upload to measure it absolutely.

## Strategy

- [ ] **Find the ceiling first.** Compute "perfect decisions on my candidates". If it's below the leaders, no model will
      get you there. Fix recall.
- [ ] **Decompose the error** (never found / false merges / missed among candidates) and work on the biggest bucket.
- [ ] **Cheap model everywhere, expensive model on the uncertain band.** That is how a 278 M transformer became
      affordable on 44 M pairs.
- [ ] **Keep a written "rejected" list** with evidence. It is the most underrated document in the repo.

## Engineering

- [ ] **Cache every stage to disk, named by its settings**, and write files atomically. Power cuts and OOM kills will
      happen.
- [ ] **Assert your assumptions in code**: no leakage, full coverage, no other rows changed. Silent failures cost more
      than crashes.
- [ ] **Pin library versions** that affect model outputs (here `transformers==4.57.6`).
- [ ] **Verify external jobs used the inputs you think** (row counts after a Kaggle dataset update).
- [ ] **Budget compute like money.** GPU quotas, RAM peaks and disk space shaped every plan. Write them on the workboard.
- [ ] **One command to reproduce** (`reproduce.py`). The final package is reviewed, and you will need it yourself at 2 a.m.

## Team

- [ ] **Tracks with single owners** on a shared workboard. Two people (or two AI sessions) never edit the same thing.
- [ ] **Log every upload** with its tag, change and score.
- [ ] **Write a manual runbook** for when the person or assistant who knows the pipeline isn't available.
- [ ] **Decide the final pick by CV** where CV is trustworthy, and use the public leaderboard only for what CV can't see.

---

<sub>← [The journey](journey.md) · [Home](../README.md) · [Code walkthrough](code-walkthrough.md) →</sub>
