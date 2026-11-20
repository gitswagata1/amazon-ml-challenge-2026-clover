# Workboard — who runs what (read before starting; `git pull` first)

Two work sessions share this repo. Each track has ONE owner. Only touch your own track's files;
log every result (CV macro F0.5) in `SUBMISSIONS.md`; only the user uploads to the leaderboard (5/day).

## Retriever go/no-go (27 Sep 02:54 IST): v9 recall 0.9776 (matcher S1), 163,520 misses; fine-tuned retriever v1 top-5 finds 34.7% (+4.8% pairs), **top-10 65.4% (+24.4%) → recall 0.9922**, top-20 78.1% (+69%). User chose top-10.

## France diagnosis + probe plan (main agent, 27 Sep 05:30 IST) — France is the whole LB gap
- **Exact LB** (user, in upload order): v8@0.72 **0.973867** · v8@0.76 0.973972 · v8.1 0.973785 · v8@0.68 0.973633 · v8.1+ce2 **0.974716**.
  With US/India LB = CV (0.9795 at the test mix): **France ≈ 0.942**. ce2 ≈ +0.002 on France (most of its +0.0009 is US/India);
  LB rises with the threshold (0.68→0.76 +0.00034, all France) → France over-accepts at the margin; tfw (v8.1) −0.0017 on France.
- **US/India generalise**: test profile = train (band pairs/S1 India 0.13 vs 0.12, US 0.21 vs 0.21; accepts, no-match, owner conflicts equal).
  Test's extra decoys (5.75 vs 4.68 S2/S3 per S1): simulated by dropping 19–35% of train S1 → −0.0001 only.
- **France**: 4× band pairs (0.56/S1), 10–20× owner conflicts; 15 cities, names = city + generic word; co-located look-alike S1 10–40× more.
  Generator (from train labels): decoys = S1 name + one list word (US group/holdings/…, India industries/…, **France: france, groupe,
  developpement, holding, participations, international, distribution**) at a nearby house number → model already rejects (p≈0).
  Noise = last word replaced by a list word (US/India center/services/service/partners: 97% matches; any other swap ≈0%).
  Records identical to another S1 (79k French pairs) → model p≈0 (right). Department vs region, sibling-orphan clusters: not causes.
- **Open group G** (`experiments/france_probe.py`; work9/g_test.parquet): same address, ONE non-typo word swapped, no twin record, no S1
  with that name: France 113,605 pairs (0.44/S1), model accepts 23%. True-match rate in train **US 0.80 / India 0.38** → France unknown;
  label-free estimate failed its train check → **LB probe**. Break-even precision for F0.5 ≈ 0.72.
- **Uploads left: 3 (final day).** Probe 1 `output_probe_g_rej` = v8.1+ce2 minus 25,105 accepted French G pairs (23,450 French S1),
  validated --check-ids. > 0.974716 → reject G in the final (+ probe 2: stricter French threshold); < → probe 2: accept more of G.
  Final = v10 + ce2 (+ ce3 if CV gains) + the probed France rules.
- **User rule (27 Sep ~06:00 IST): if anything is rebuilt/retrained (v10 rerun, v11, France variant, any new retrain_extra), DROP the word TF-IDF columns (tfw_core/tfw_addr, `--extra .../tfw_{split}.parquet`).** Evidence: v8.1 (+tfw) LB −0.000082 vs v8 despite CV +0.0002 → France ≈ −0.0017. Running v9/v10 keep it (no retrain just for this). Char TF-IDF stays.
- **PROBE RESULT (27 Sep ~06:45 IST): France G common<->common rejection = 0.975834 vs base 0.974716 (+0.001118 LB, France ≈ +0.0075) → IN THE FINAL.** Whole-group rejection NOT used (probe1_sim: rare-word swaps are mostly true). Unseen-country simulation (probe1_sim.py) matching France = US→India WITHOUT ce (+0.006 predicted) → use it to screen further French rules. Exact LB list: v8 0.973867, v8@0.76 0.973972, v8.1 0.973785, v8@0.68 0.973633, v8.1+ce2 0.974716, probe 0.975834. Uploads left: 2.
- **P2' (same sibling-word swap at a DIFFERENT address, symmetric words only: france_rule.py --address different --lr-max 0.5)**: rule_sim.py (exact rules, P1 then P2'): US→India no-ce +0.0094 (removed precision 0.05) · US→India ce +0.0003 · India→US no-ce +0.0004 · India→US ce −0.0001 (P1 in the same arms: +0.0059/−0.0018/−0.0006/−0.0013). Probe file output_probe_p2_rej = best upload (0.975834) − 8,529 French pairs. Rejected in screening: P4 (+1 common word: French accepts are legal form 'sci' + generator noise words), P5 (−1 word, −0.010), P3, P6 (tiny).
- Capacity (tree count): 300k/3-fold 1202 trees 0.98558 vs 2404 0.98550 → no gain; laptop re-runs 600k/5-fold curve (amizon-capacity600-w), ~11:30–12:00 IST.
- **27 Sep 08:51 IST: WSL restarted (OOM from my 40 GB proxy_mine.py alongside v10 features) — all jobs killed; v10 resumed 09:14 (logs/v10_resume.sh: kept files on disk, Kaggle kernels reused, NO tfw per user rule).** US/India conflict rule on the full v9+ce2 decision: +0.00018 both countries @δ0.2 → IN v11 (US/India only).
- **ce3 retrained properly for v11 (user): 4 e5-large shards, one full epoch each on disjoint quarters of ce_train_h0 (kernels amizon-ce3-s0/s1-w kaggle-B, s2/s3-w kaggle-A), ETA ~14:30 IST; adopt if the averaged score beats ce2 by ≥ +0.0003 (half-1 stage 2). v10 continues without ce3. Plan kept in the plan file.**
- **v10 GPU race (found 09:46 IST):** kaggle-B's ce/embed-train kernels (pushed 8 s after the dataset upload) scored the OLD amizon-miss-train-w version (v9's 5.06M pairs), covering 0 of v10's 12,229,475 new train pairs; test scores are complete. User: the RTX 3050 re-scores them (dataset kaggle-A/amizon-miss-train-w → out kaggle-A/amizon-v10-train-gpu-w); logs/v10_finish.sh waits, merges with coverage asserts, runs retrain_extra (3 seeds, no tfw). Lesson: after a dataset version update, verify the kernel mounted it (row count) before trusting outputs.
- **Power cut 27 Sep ~10:38 IST (workstation off):** checked after reboot — all work10 files intact (train_features 54,986,629 rows, test_features 44 parts / 43,905,439 rows, rarity, emb_in, miss_train, test GPU scores); git clean at 6b0291e. Killed: v10_finish waiter, status loop, monitors (restarted 10:41). Unaffected: 4 ce3 shards on Kaggle (RUNNING), RTX 3050 scoring, old laptop.
- **27 Sep 12:35 IST: v10 training started** (train GPU scores from new Kaggle account kaggle-C, coverage-checked; RTX 3050 copy not needed). **P5 France-adapted CE (organisers: self-training + synthetic pairs from the provided records are allowed):** experiments/fr_synth.py → 885,573 French pairs (300k real positives p≥0.995 winners outside the France-rule lists, synthetic sibling-word swaps, decoy words at nearby numbers, P2' + easy negatives) → fine-tune ce2 h0 (e5-base) 1 epoch on kaggle-C (kernel amizon-frce-w, experiments/kaggle/frce_train.py) → scores 1,060,401 French pairs (v8.1+ce2 and v9+ce2 universes). Plan: upload #1 = France-only probe on the 0.975834 base.
- **P5 France probe LB 0.977522 (+0.001688 vs 0.975834) → P5 in the final.** ce3 (proper, 4 shards) +0.00011 → not adopted. v10 CV 0.9880 (US 0.98757, India 0.98871). kaggle-A hit its 30 h weekly GPU quota (15:14); kaggle-C is the GPU account now.
- **LB A/B (27 Sep ~16:40): A (v9+ce2+P5+P1+P2'+conflict) 0.983893 = best; B (A + ce3) 0.983474 → ce3 −0.000419 on the LB (dilutes P5 on France) → REJECTED. France ≈ 0.966 backed out of A. 1 upload left = v11 final.**
- **v11 final build (27 Sep ~17:00 IST):** v10 test-band ce2 scores from the RTX 3050 (dataset kaggle-A/amizon-v10bandte-ce2-w, 1,484,423 pairs per half model, exact coverage; vs Kaggle-scored v9 band on 802,741 shared pairs corr 1.000000, p99 |d| 0.002 → transformers pin OK) → work10/band/h{0,1}/test_ce.parquet. P5 = work9/frce_out (1,060,401) + CPU-scored work10/frce_miss (210,562) = 100% of v10's French pairs p ≥ 0.02 (1,221,454; text idx verified). Build: `logs/v11_build.sh` (merge P5 → ce2_band.py --frce → final_assemble.sh … output_final_v10 0 → validator). kaggle-D kernels amizon-v10g-h0/h1-w (same scores) still QUEUED, no longer needed.
- **v11 READY (27 Sep 17:09 IST):** `logs/v11_build.sh` → stage 2 CV 0.98876 @ 0.68 (+0.00074 vs v10 0.98802), P5 100% of French pairs, conflict 6,788 / P1 11,947 / P2' 229 (work9-prep lists; work10-prep lists would change 2,964 French rows), validator PASS → `output_final_v10`, staged at Downloads\upload_FINAL_v11 (sha256 matching 8d9be3db…, candidates 97454074…). handoff4 (review branch): B1 kept (A is LB-validated with ce2_band --frce), B2 fixed (CPU-scored misses), B3 offset 0 default, B4 work9-prep lists, B5/B6 moot, B7 P2' = 229 pairs. B2 follow-up (cloud, 57b879c): CPU vs Kaggle P5 logits on 2,000 Kaggle-scored pairs: pearson 1.0, max |Δ| 0.040 (< 0.05) → consistent.
- **v12 (plan: the plan file, 27 Sep 17:50 IST; 1 upload left, best counts):** line 0.989 needs France ≈ 0.989 (+0.02, 2× P5) → not reachable with measured levers; maximise expected LB. **L1 DONE:** stage-2 OOF (ce2_band --save-oof) + experiments/country_thr_cv.py: with the conflict rule the CV-optimal US/India threshold is 0.58 (flat 0.56–0.62), not 0.68: CV 0.98903 → 0.98921 (+0.00018; India 0.98995→0.99012, US 0.98842→0.98860); per-country thresholds/deltas add nothing more. France kept at 0.68 (offset +0.10). `output_final_v11L1` (15,007 US/India rows vs v11, France identical), validator PASS, staged Downloads\upload_v11L1_fallback = new fallback. **L2 RUNNING:** P5b = ce2 h1 fine-tuned on P5's 885,573 French pairs (Kaggle kaggle-C/amizon-frce-h1b-w, 17:48, ETA ~19:00), scores all 1,270,963 frce pairs; v12 = stage 2 with P5+P5b averaged (`logs/v12_build.sh`). L3 (self-training round 2) dropped: round 2 labels the model's confident French decisions, which the P1 probe showed are ~70% wrong in look-alike groups.
- **27 Sep 19:10 IST (5 uploads granted, 2 left):** LB v11+L1 0.98533, v11/work10-rules 0.985084, **hybrid (v11+L1 US/India + A's French rows, experiments/country_merge.py) 0.985926 = best** → v9 is better on France (+0.004), v10 on US/India. P5b (ce2 h1 fine-tuned on P5's data, kaggle-C/amizon-frce-h1b-w) done: corr with P5 0.9867, sign agreement 98.2%. **Upload #4 = hybrid + P5b** (A's France recipe with P5+P5b averaged; 3,272 French rows differ from the best) staged Downloads\upload4_hybrid_P5b. v9/v10 France ensemble dropped (logit corr 0.9969 on shared pairs, 3,725 rows, expected ≈ 0). **L3 running:** fr_synth round 2 from A's P5-improved probs (889,946 pairs) → kaggle-C/amizon-frce-r2-w (queued 19:05) → logs/r2_build.sh builds #5 candidates hyb3a (P5+P5b+r2) / hyb3b (P5+r2).
- **FINAL (27 Sep 22:50 IST): `upload21_hybrid_P5b_France091` = output_probe_hyb2_p5b_frt0.19 = LB 0.986371** (France threshold curve: 0.88 0.986335, 0.91 0.986371, 0.92 0.986368, 0.93 0.986345; flat peak). Previous: 0.92 = 0.986368. Diagnostic (French rows empty) 0.849872 → US/India LB 0.9897 (≥ CV), France 0.9673 = the whole gap. Last-hour checks, all negative: row-aware France rule 0.986298, P5-tail removal f = 0.008, P5 rescue of French pairs with stage-1 p < 0.02 (12k sample: 0.25% logit ≥ 0, none ≥ 7), exact-name same-address copies missed by blocking: 38, twins/S1-sibling/acronym categories ≤ 500 pairs, US/India row-size thresholds +0.000015 CV, count-based conflict −0.0008 CV. France's missing copies are noisy variants neither blocking, stage 1 nor P5 recognises → needs a new French-aware matcher (not feasible before the deadline).
- F3 (user OK): **ce3** = e5-large band cross-encoder, half 0 only, 1 kaggle-A slot (laptop), stage-2 on half 1.

## MAIN AGENT = workstation Track D session (user, 27 Sep ~02:45 IST). Assignments
| Resource | Task now | Next |
|---|---|---|
| **v10 (launched 27 Sep ~03:20 IST)** | `logs/v10_chain.sh` (PID 572400; waits for v9's training PID 551820): branch `v10` 81eb341 (review D1+D3, **no D2**; `--knn2 work/ret --knn2-top 10` fine-tuned retriever v1; `--tfidf-all`; 3 seeds) in `work10` → CPU features ‖ GPU (train: kaggle-B, test: kaggle-A) → `output_v10` | ETA ~11:00–12:00 IST |
| Workstation (main agent) | v9 training (CV ~03:15, output ~03:45 IST); `key_audit.py` on work9 (review branch, nice 15) | ~04:00: **v10 rebuild** in `work10` from branch `v10` (98271ba = main + review branch D1–D7): D1–D4 if the audit passes, + retriever candidates if the go/no-go passes, + `--tfidf-all`; parallel chain like v9 |
| Laptop (Entity matching…, Remote Control) | L1 packaging/docs for v9 (README flags, Documentation, rejected list) | L2 keep kaggle-A's GPUs free for v10 scoring (~06:00–08:00) |
| recall-diag session (workstation) | `france_gauge.py` on v8 / v8.1 / v9 outputs (label-free France signal) | idle during the v10 build |
| Backup laptop (RTX 3050) | **offline**; standby GPU | retriever v2 (D7 mask) only if needed after v10 |
| Kaggle kaggle-A | amizon-rets-train (retriever search, running) | v10 GPU scoring |
| Kaggle kaggle-B | free | v10 GPU scoring (main agent) |
| Kaggle kaggle-E | no GPU (phone not verified) | CPU only / unused |

## Manual runbook if no session is available (workstation, WSL: `cd ~/Amizon`)
1. **Is v9 done?** `tail -5 logs/v9_post2.log` → last line "v9 done" + "CV macro F0.5 = …". Progress: `tail logs/v9_cpu.log logs/v9_gpu.log logs/train_v9.log`.
   (Or on GitHub: branch `status-new-machine`, file `STATUS_NEW_MACHINE.md`.) If a line says FAILED, stop and wait for the owner.
2. **Compare:** upload v9 only if its CV > **0.9806** (v8.1) / 0.9804 (v8).
3. **Validate:** `python3 utils/validate_submission.py --matching output_v9/matching_results.tsv --candidate output_v9/candidate_pairs.tsv --test-dir dataset/test --check-ids` → must print PASS.
4. **Upload** `<repo>\output_v9\matching_results.tsv` on the portal (one login only). Note the LB score in SUBMISSIONS.md.
5. Retriever go/no-go: `cat logs/ret_gonogo2.log` (go = ≥ 50% of misses at ≤ +10% pairs). Building v10 is not automated.
6. Fallback if v9 fails: `output_v81/` (CV 0.9806, validated) or `output_t072/` (v8, LB 0.974).
7. Power cut: see "recovery after a power cut" below; the Kaggle kernels keep running regardless.

## FINAL PLAN (agreed 26 Sep 17:00 IST: laptop + Track D session + user). Deadline 27 Sep 23:59 IST; final upload ~21:00
**Current best: v8 = CV 0.9804 @ 0.72, public LB 0.974** (rank was 1069 with v7 0.953; top 0.989).
Rule: every ingredient needs its own passed test; a version replaces v8 only if its CV is higher; final pick by CV.

| Stage | What | Owner | Time | Status |
|---|---|---|---|---|
| **1. v8.1** (no new pairs) | v8 + word TF-IDF (tfw_core/tfw_addr, **+0.0012** measured) + any F1–F6 winner (> +0.001 over fctl 0.96128), retrain from saved features | workstation | ~1.5 h, start ≤ 19:30 IST | **v8.1 CV 0.9806** @0.72 (v8 + tfw only; `output_v81/` 5,708,474 matches, **not yet validated**). F1–F6 vs fctl 0.96128: F1 +0.00085 (best), F6 +0.0002, F2–F5 ≈ 0; none ≥ +0.001 alone; fALL pending (~21:30 IST) |
| **2. v9** = ONE candidate rebuild overnight | legal-word fix 472a35c (bug fix) + **dictionary A** (if its CV test passes) + **bidirectional rescue B** (if passes) + **N4 kNN candidates** (if extra pairs/S1 at top-10 are acceptable) + stage-1 features → score NEW pairs only on both Kaggle accounts → train + predict | workstation (+ Kaggle both accounts) | ~7 h: caches+pairs ~3 h, GPU ~1.5 h, train+predict ~3 h → **~01:00–03:00 IST** | go/no-go at ~19:30 IST |
| — N4 search (input to v9) | e5-small top-20 same-country neighbours → `{split}_knn.parquet` (i, j, knn_rank, knn_cos); **use top-5** (+3.2% pairs on test; recovers 18.5% of v8's train misses) + knn_rank/knn_cos features | laptop → Kaggle `amizon-n4s-train/-test` | done 18:15 | both outputs ready |
| — A test (dictionary, name features only) | `experiments/a_dict_ab.py`, A/B table, ctrl vs dict | laptop | done 18:15 | **PASS: 0.95286 → 0.95381 (+0.0010)** → in v9 via `prep.translate` + `native_dict.tsv` (013622c) |
| — B test (bidirectional rescue) | `experiments/b_rescue_ab.py`, A/B table, top-20 vs + reverse-rank 1 | laptop | done 17:05 | **PASS: 0.95182 → 0.95278 (+0.0010)** at +0.53% pairs → in v9 |
| **3. ce2 add-on** (+0.0005 measured) | band published: `amizon-v8-band-h0/h1` (train 493,829/494,027 + test 950,869); laptop kernels `amizon-v8band-h0/h1` (score-only) queued behind the N4 search on kaggle-A; 2nd-stage [logit p, ce2] | workstation + laptop | ~1.5 h | kernels ready, push ~18:30 IST |
| **5. Re-tune LightGBM** | Optuna on Kaggle CPU, warm start = best_params | laptop | done 23:15 | **no gain**: 0.95993 → 0.95996 / 0.96000 (noise) → keep best_params.json |
| **6. k = 15 vs 20 check** (re-added by user 17:05) | on v9: filter pairs by brank ≤ 15, one CV run; take k=15 if it loses < 0.0005 (organisers rank smaller candidate sets higher) | workstation | ~1.5 h, 27 Sep morning | |
| **4. 27 Sep** | pick by CV → 3-seed final fit (if time) → docs (2 pages), README flags, docstrings → zip from tag → **upload ~21:00** | all | | |

**Not doing** (tested and rejected, or no time): pseudo-labelling (−0.036), N1 context stacking, N5 empty-address block, N6 voice-neutral skeleton, ce2 replacing ce, full blocking rebuild beyond v9, CatBoost/blends, v4/v5, chaining.
Test findings behind v9: v8 misses 262,880 true pairs (3.44%); 52% have an EMPTY candidate address (unrecoverable); native-script words come from a fixed vocabulary (2,000 words = 98.6%), mined dictionary 0% → 79% word match (out of sample); N4 top-10 recovers 30% of v8's misses (~50% of the non-empty-address ones: domain names 71%, typos 53%, native script 32%).

## v9 build (workstation, started 26 Sep 12:51 UTC) — exact recipe
- **Status 21:15 IST:** phase 1 (work9 caches) done 13:01 UTC; the chain deadlocked (its `pgrep -f retrain_extra.py` matched its own shell) 14:34–15:45 UTC; user approved unblocking; **phase 2 pairs running since 15:45 UTC** (PID 334156, `logs/pairs_v9.log`) → GPU on new pairs → train + predict: **v9 ready ~03:00–04:30 IST**.
- Lesson for all scripts: never wait on `pgrep -f <name>` from a shell whose own command line contains `<name>`; wait on a PID or a done-file.
- prep = main (fr-norm + legal-word fix 472a35c + native dictionary 013622c, `native_dict.tsv`); caches in **`work9/`** (v8/v8.1 keep `work/`).
- Pairs: `run.py pairs --work work9 --n-train 2206821 --k-use 20 --pool 1500 --addr-partial --addr-keys --rescue --knn work/n4 --knn-top 5 --exclude-s1 work/ce/ce_train.parquet`
  → `work9/{train,test}_pairs_k30_p1500_ap_ak_use20_rs_knn5.parquet` (+ knn_rank/knn_cos features). Chain: `logs/v9_chain.sh` (phase 2 waits for v8.1 to exit: blocking peaks ~36 GB).
- N4 on ALL 262,880 v8 misses (train): top-5 recovers 17.5% at +4.5% train pairs (+3.2% test), precision of new pairs 2.3%.
- 26 Sep ~16:00: + **1a rare-empty candidates** (`--rare-empty`) + **1b rarity features** (`experiments/rarity_cols.py` → `work9/rarity_{split}.parquet`,
  +0.0038 on the A/B table, + scale-free flags name_df_s1_eq1 / name_df_cand_le1). F1–F6: none (all ≤ +0.001).
- Automated after blocking: **`logs/v9_post2.sh` (PARALLEL, replaces v9_post.sh at 18:00 UTC)**: rare-empty rerun → [CPU: tfw + rarity columns + `run.py train/predict --features-only`] ‖ [GPU: new-pair scores] → `retrain_extra.py --work work9` joins GPU + extra columns → CV, fit, predict → `output_v9`. Old description, for reference: `logs/v9_post.sh` (waits on PID 334156): pairs rerun with `--rare-empty` → tfw + rarity columns →
  NEW-pair GPU scores (kaggle-B `amizon-{ce,embed}-{train,test}-w`, datasets `amizon-miss-{train,test}-w`) → merge into `work9/gpu` →
  `run.py all ... --emb-dir work9/gpu --extra 'work9/tfw_{split}.parquet' --extra 'work9/rarity_{split}.parquet' --save-features --save-test-features --out output_v9`.
  Log: `logs/v9_post.log`; after a power cut, rerun the remaining steps of that script by hand (pairs/columns are cached).
- CV reading: leak check done: the all-train dictionary scores 0.95378 vs 0.95381 leak-free → no correction needed.
  Pick v9 over v8.1 if its CV is higher by > ~0.001.

## Path toward 0.99 — analysis (26 Sep ~23:00 IST, `experiments/err_decomp.py` on v8 OOF)
| v8 (CV / train OOF) | all | India | US |
|---|---|---|---|
| actual | 0.9804 | 0.9768 | 0.9828 |
| **ceiling = perfect decisions on v8's candidates** | **0.9890** | 0.9838 | 0.9925 |
| no false positives | 0.9833 | 0.9796 | 0.9858 |
| no missed matches among candidates | 0.9862 | 0.9810 | 0.9896 |
- Loss 0.0196 = candidates never found **0.011 (56%)** + matcher on candidates 0.0086 (FN 0.0058 > FP 0.0029).
- By S1 type: **"partly right" multi-match S1 = 0.0150 (76%)** (avg 3.5 true matches per S1); fully missed 0.0025; no-prediction 0.0014; singleton FP 0.0007 (singletons 5.6%).
- **France** (backed out of LB 0.974, assuming US/India score their CV): **≈ 0.943** vs 0.9795 for US/India at the test mix → costs ≈ 0.0055 LB.
  Label-free: France's predicted-match profile is normal (94.5% of S1 with ≥ 1 match, 3.3 per S1 = train), but 3× more S1 have an
  uncertain top candidate (0.5–0.9: 1.0% vs 0.3%), and 62% of its accepted prob<0.95 matches have a word conflict (US 29%):
  French **sibling businesses** at the same address differing by one generic word ("Chti Parents" vs "Chti Centre", p=0.87 accepted).
- 0.99 needs CV ≈ 0.995 + France parity; v8's candidate ceiling is 0.989, so candidates (v9) come first. Realistic by the deadline: ~0.982–0.985.

- **User, 26 Sep 23:30 IST: top-10 LB teams are ABOVE 0.99** → our v8 candidate ceiling (0.989) is the binding limit; others have near-complete recall.
- Structure checks (train): matches per matched S1 mean 3.67 (median 4, max 11); 26% of candidates are unmatched decoys (1.34M per source);
  S2/S3 split irregular (13% of entities have no S2 copy, 12% no S3) → no count prior; **no row-order leak** (consecutive-row share 0).
- v8 misses (262,880): 97% belong to an S1 with other matches found; 51.5% have an empty address; only **15% sit in ranks 21–30**, 85% are
  beyond rank 30 / never blocked → widening k barely helps. Exact-key sibling expansion (name/core of a PREDICTED match, df ≤ 3–10):
  precision 1–4% (look-alike siblings), empty-address-only 23–48% precision but recovers 4–7% of misses → small.
- Off-the-shelf e5 kNN top-10 recovers 30% of misses; a **fine-tuned retriever** is the untried big-recall lever (see plan).

### Push toward 0.989 (user-approved 26 Sep ~23:45 IST)
| # | Item | Owner | When (IST) | Bar / expected |
|---|---|---|---|---|
| P1 | v9 (running, automatic) | workstation | ~04:30–05:30 | LB ~0.980–0.982; upload if CV > 0.9804 |
| P2 | **France transfer**: F1/F4/F3 + new `cand_core_is_other_s1`, per-split word stats; LOCO both directions | workstation, other session (nice 19) | tonight | +≥0.002 on the unseen country, no pooled loss > 0.0005 → up to +0.005 LB (France ≈ 0.943 now) |
| P3 | **Fine-tuned e5-small retriever**, trained only on the CE's 99,948 excluded S1 (no leak) → top-20 kNN train+test | laptop → Kaggle | model ~02:00, kNN ~03:30 | go if ≥ 50% of v9's remaining misses at ≤ +10% pairs |
| P4 | Anchor features (similarity to the S1's top-OOF candidate) | laptop A/B | tonight | ≥ +0.001 |
| P5 | v9.1 = v9 + P2/P4 winners, retrained from v9's saved features (`retrain_extra.py --work work9`) | workstation | ~07:00–09:00 | |
| P6 | v10 = + P3 candidates (pairs rerun + GPU for new pairs + train/predict) | workstation + Kaggle | ~11:00–13:00 | if P3 passes |
| P7 | Final pick by CV + France LB evidence (uploads: v9, v9.1 ± France features, v10), package | all | by 21:00 | |

## Rejected or not adopted — with reasons (all sessions; don't redo without new evidence)
| Idea | Evidence | Reason |
|---|---|---|
| v4: candidate-competition + cluster-support features | CV 0.9485 vs 0.9487 | no gain |
| v5/v5b: learned LambdaRank re-ranker | recall 94.5% but CV 0.9440 / 0.9444 | pulls in harder look-alike negatives the matcher can't separate |
| Pair chaining (transitive S2–S3 links) | CV no gain | chains propagate false merges; F0.5 punishes them |
| More LightGBM trees (2× / 3× at the same lr) | 300k 3-fold 0.98550 vs 0.98558; curve 300k 5-fold 0.98558/0.98554/0.98551; **600k 3-fold 0.98581/0.98581/0.98579** | flat even at 2× data: keep best_params (1,202 trees) |
| Per-S1 decision rules incl. expected-F0.5 set selection | 70 variants | none beat one global threshold + one-owner rule |
| k = 30 candidates per S1 | 0.9489 vs k=20 0.9487 | same score, 33% more pairs; organisers rank smaller sets higher (k=10: 0.9468, too costly) |
| CatBoost; LightGBM + CatBoost blends | 0.95048; best blend 0.95268 | below tuned LightGBM 0.95287 |
| Untuned LightGBM params | 0.95008 | Optuna-tuned +0.0028 (full 400k check; the 150k/3-fold estimate +0.0002 was noise) |
| Blocking A (old) / B (pool 1500 + partial re-rank only) | top-20 recall 92.6% / 94.2%; CV 0.9492 | superseded by blocking C (96.6%, CV 0.9669) |
| Higher blocking key caps (30k / 50k) | no recall gain | common keys add noise, not matches |
| Full embedding-search blocking (instead of keys) | after address keys only 0.54% of true pairs share no key; ceiling +0.002 | too costly for the ceiling. *N4 revisits it only as a small top-k add-on* |
| France robustness: drop / normalise country-skewed blocking features (b_rel_s1 …) | LOCO ablation −0.0026 to −0.0067 | the country signal in them is real matching signal, not a France risk |
| Pseudo-labelling, LightGBM matcher | US→India sim 0.821 → 0.820–0.821 | pseudo-labels are the easy pairs; nothing new to learn (allowed by organisers, still no gain) |
| Pseudo-labelling, cross-encoder | −0.036 / −0.038 (India sim) | CE drifts toward LightGBM's view; a CE that never saw a country already transfers (+0.053 there) |
| ce2 (e5-base OOF CE) as a full feature | stack +0.0005 (0.96103 vs 0.96049) | below the +0.002 bar for scoring all test pairs with two e5-base models; kept only as the cheap band add-on |
| More epochs of the e5-small CE | – | superseded by ce2: even a bigger CE gave only +0.0005 |
| N1: 2nd-stage list-context model (rank / gap / counts) | 0.95960 vs 0.96005 | no gain |
| N5: empty-address same-name block | recovers 13.5–16.4% of misses, precision 1.6–2.6% | same name + no address is unresolvable; floods candidates |
| N6: voice-neutral skeleton (g≈k, b≈p, d≈t) for Indic names | same-key share 0.107 → 0.116 | too small |
| "St X" (space) → Saint for French | 0.04% of French vs 0.53% of US addresses | would damage 13× more US addresses; hyphenated St-/Ste- (97% of cases) already fixed |
| Word statistics / TF-IDF fitted on train only for test-time features | – | train has no France → every French word unknown; fit on train+test text (label-free) instead |
| CE feature with all-data training, no exclusion | – | leakage: CE saw 99,948 S1's labels → `--exclude-s1` |
| --block-sample CVs as absolute numbers | 0.9669 vs full 0.9804 | block-sample weakens b_n_c / b_max_c; use only for relative A/B |
| Local T1000 GPU for e5 / CE | 2% GPU use, WSL GPU bridge crashes | moved to Kaggle |
| AWS instead of Kaggle for scoring | – | GPU quota uncertain (often 0 on new accounts); Kaggle proven and free |
| 3 threshold probes | – | reduced to 2 (0.72 / 0.76): uploads are scarce; LOCO says threshold transfer costs ≤ 0.0012 |
| F1–F6 matcher features (word IDF, CE rank/gap, decoration, typo shape, script flags, address parts) | vs fctl 0.96128: +0.00085 / +0.0002 / +0.00014 / +0.00009 / +0.00002 / +0.00002; all +0.00094 | none clears the +0.001 noise bar |
| Idea 2: importance weighting toward France (domain-classifier weights) | LOCO US→India w −0.0057 / sqrt +0.0016; India→US −0.0010 / +0.0006 (bar +0.003) | countries separate almost perfectly (98% of rows clipped at 0.2), so the weights carry little signal |
| P2 France-transfer features (F1/F3/F4 + sibling-name feature), LOCO | best +0.0015 one direction, negative the other (bar +0.002 both-safe) | sibling signal already in b_n_c/b_max_c + one-owner; CE is what transfers |
| P4 Anchor features (similarity to the S1's top-OOF candidate) | +0.00047 (bar +0.001) | no gain |
| Optuna re-tune #2 | 0.95993 → 0.96000 | noise; keep best_params.json |
| Review D2: house-number keys from split comma parts (cloud session, a review branch) | key_audit on real work9: true pairs with shared h: key India 0.755→0.673, US 0.777→0.746; 69 gained vs 203 lost every shared key per 200k | lowers blocking reach on US/India |
| Review D1/D3/D4/D7 (French norm v2, leet digits, postcode parse, retriever mask) | deferred, not rejected | need a rebuild or retrain; no time / unmeasurable for France. Status in HANDOFF_REVIEW.md §6 on that branch |
| Review handoff2 P2: conflict-aware acceptance (reject a winner whose record's 2nd S1 has p >= 0.5 within a gap δ) | in-country v9 OOF δ 0.1/0.2/0.3: US +0.00012/+0.00017/+0.00018, India +0.00015/+0.00016/+0.00013; unseen US→ALL India (no ce) ON TOP OF P1+P2': +0.00001 / −0.00006 / −0.00019 (removed precision 0.65–0.71) | P1+P2' already remove the sibling conflicts; what is left is near break-even → not adopted (`experiments/conflict_sim.py`) |
| ce3 = e5-large band cross-encoder (half 0, 25% of ce2-h0 pairs, time cap hit at 72% of the epoch) | half-1 S1: A3 [lo, ce2, ce3] 0.98697 vs A2 [lo, ce2] 0.98693 = +0.00004 (bar +0.0003) | no gain over ce2; a full run needs ~12 h GPU → not retried |
| Richer ce2 stage 2 (A + 7 pair features) | 0.98712 vs 0.98706 = +0.00006 (bar +0.0003) | the main model already uses them |
| 2x / 3x trees (300k S1, 5-fold curve) | 1202: 0.98558, 2404: 0.98554, 3606: 0.98551 | flat; 600k/3-fold check pending (laptop) |
| Retriever v2 (rets2, retrained) vs v1 — skipped by v10's gate timing bug, evaluated 27 Sep ~09:45 IST | v9 misses found: top-5 34.6% vs 34.7%, top-10 65.4% vs 65.4%, top-20 78.0% vs 78.1% (same pair cost) | identical to v1 → v10 keeps v1; nothing lost by the skip |
| Unseen-country corrector (LightGBM on label-free pair features: logit p, 2nd-best S1, claims, name x address relation, rarity; trained on one no-ce proxy direction, tested on the other) | held-out US: best −0.006 (P1+P2' −0.0001); held-out India: best −0.003 (P1+P2' +0.0148) | learns the training country's own quirks; the generator-logic rules transfer, a learned corrector does not |
| ce3 PROPER: 4 e5-large shards, one full epoch each on disjoint quarters of ce_train_h0 (all 3.95M pairs), scores averaged | half-1 stage 2: A3 [lo, ce2, ce3] 0.98704 vs A2 [lo, ce2] 0.98693 = +0.00011 (bar +0.0003) | e5-large adds ~nothing over e5-base ce2; not worth ~1.5 GPU-h for v10's band |
| External data / lookups, non-MIT/Apache or > 8B models, second Kaggle account of one person | – | challenge rules / Kaggle ToS |

## Laptop → main agent (27 Sep 05:12 IST; cross-session route lost after the laptop's re-login)
- **L3 retriever v2:** `kaggle-A/amizon-rets2-train` + `amizon-rets2-test` **COMPLETE** (04:58) → `{split}_knn.parquet` ready for the v2-vs-v1 gate.
- **L4 v9 ce2 band:** `kaggle-A/amizon-v9band-h0` **COMPLETE** (log: train 444,128 + test 896,960 scored = the dataset exactly); `amizon-v9band-h1` **COMPLETE** 05:13 (train 444,749 + test 896,960 scored = the dataset exactly). Outputs `train_ce.parquet` + `test_ce.parquet`; test ce2 = mean(h0, h1).
- kaggle-A's GPU slots free by ~05:30 IST. Laptop is ready for the v10 band (post the dataset names here).

## Final-branch procedure (review F11; dry run by the laptop 27 Sep 10:15 IST — tested, NOT executed)
Run on the workstation after v10 (and its ce2 band / France rules) finish and the final output is validated:
```bash
git fetch origin && git checkout -b final origin/main
git merge --no-ff origin/v10 -m "final: main + v10"
#   dry run: exactly ONE conflict, experiments/retrain_extra.py (both sides only changed the zip-layout import path);
#   take v10's hunk = the code that trained v10:
git checkout --theirs experiments/retrain_extra.py && git add experiments/retrain_extra.py && git commit --no-edit
python code/business_entity_resolution/src/prep.py          # self-test must print "ok"
git diff --quiet origin/v10 -- code/business_entity_resolution/src && echo "src == v10"   # dry run: YES
git tag -a v10-final -m "final submission: <CV>, <LB>" && git push origin final v10-final
./make_zip.sh output v10-final                               # output/ = the validated final folder
```
Dry-run results: merged src/ identical to v10; README/docs/experiments from main; zip from the merged commit = 41 files;
from the extracted zip, `--help` OK for retrain_extra, ce2_band, band_export, country_threshold, france_rule, france_probe.
If main gets new commits before the merge, re-check the conflict list (`git merge --no-commit` first).

## Machines
| Machine | Owner of tracks | Kaggle names it may use |
|---|---|---|
| Old laptop (Linux, 16 cores, 16 GB; **disk ~4 GB free** → coordinator only: Kaggle jobs, small A/Bs, docs/zip) | A, B, E | `amizon-ce`, `amizon-embed`, `amizon-tune`, datasets `amizon-*` |
| New Windows machine (WSL2, 64 GB, T1000) | C, D | anything ending in `-w` (e.g. `amizon-ce-w`); v8 GPU scoring runs under teammate account `kaggle-B` (kaggle-A's 2 GPU slots = laptop's ce2) |
| Kaggle GPU / CPU | used by both, names as above | |
| AWS EC2 | down (credentials invalid since 25 Sep ~08:00) | |

## Tracks
| Track | Owner | Status (update when it changes) |
|---|---|---|
| **A. Candidate budget**: CV with top-10 / top-20 / 30 candidates per S1 (`--k-use`). Organisers: smaller candidate sets rank higher. | laptop | **done: k=20 adopted** (0.9487 vs 0.9489 at k=30, 33% fewer pairs; k=10 0.9468) — use `--k-use 20` |
| **B. GPU features**: cross-encoder (`ce.py`) + e5 embeddings (`embed.py`) on the regenerated pairs, then CV A/B | laptop + Kaggle GPU | **done: v7 CV 0.9603** (cross-encoder +0.0077); test scored on 2xT4 (`amizon-ce-fast`) |
| **C. Model blend**: CatBoost (Apache) and 50/50 LightGBM+CatBoost vs LightGBM, via `experiments/ab.py` (reference 0.95008) | new machine | **done: rejected**. CatBoost 0.95048; best blend (0.8 LGBM_tuned) 0.95268 < tuned LGBM 0.95287 |
| **D. Full-data training**: blocking C (`--pool 1500 --addr-partial --addr-keys`, ab7: CV 0.9669 vs 0.9492, --block-sample so relative only), all train S1 minus the CE's 99,948 (`--exclude-s1`), k=20, tuned LGBM, TF-IDF + emb + ce | new machine | **CV 0.9804** @ t=0.72 (folds 0.9779–0.9781 @0.5; v7 0.9603); test prediction running (auto). Next: validate, tag `v8` with prep.py from a048dc3, zip, user uploads; export uncertain pairs for the ce2 add-on |
| **E. Submissions, zip, documentation** | laptop (user uploads) | **v7 = best (LB 0.953, CV 0.9603)**; 1 upload used on 26 Sep |

## Already decided (don't redo)
- Rejected: v4 competition/cluster features; learned LambdaRank re-ranker (v5/v5b); pair chaining; per-S1
  decision rules incl. expected-F0.5 set selection (70 variants, none beat threshold + one-owner rule);
  (Optuna was first judged +0.0002 on 150k/3 folds; the full 400k/5-fold A/B shows **+0.0028** → tuned params adopted.)
- Kept: TF-IDF + e5 embedding cosines (+0.0014 CV together).
- Never: any external lookup. **Pseudo-labelling on test: ALLOWED (organisers replied 26 Sep ~09:15), but REJECTED by us**: no gain for LightGBM (0.821 → 0.820–0.821) and −0.038 for the cross-encoder in the US→India simulation (see SUBMISSIONS.md). Don't redo.

## Leakage rule for any model that uses the cross-encoder feature (`ce`) — applies to track D
The cross-encoder was fine-tuned on the candidate pairs of **100,000 training S1 records** (ids = unique `i` in
`ce_train.parquet` of Kaggle dataset `kaggle-A/amizon-ce-data`). Its scores on those records are optimistic
(it has seen their labels). **Exclude these 100k S1 records from the matcher's training/CV set** whenever `ce` is a
feature (e.g. all-data training = 2.2M − 100k ≈ 2.1M S1: `run.py train --n-train 2206821 --exclude-s1 <ce_train.parquet> ...`). v7 (600k sample) is already disjoint from them (overlap 0).
Test-time `ce` scores are unaffected.

## ce2: stronger out-of-fold cross-encoder (agreed laptop ↔ new machine, 26 Sep)
- Laptop (track B): `multilingual-e5-base` (MIT, 278M), 2 epochs. Two models on the two halves of train S1,
  `half = (i * 2654435761 % 2**32) % 2` (`ce_data.py ce_half()`). Each trains on a 200k-S1 sample (~4M pairs) of its half
  and scores the OTHER half's band pairs. Test = mean(A, B). Keep e5-small `ce` as a separate feature for the A/B.
- New machine (track D): after the first blocking-C model, export the uncertain band as a `-w` dataset with columns
  `i, j, half`. Train rows: LGBM OOF prob in [0.02, 0.98]. Test rows: predicted prob in [0.02, 0.98]. `ce2` = NaN outside the band.
- Validate on the 400k table before full use. Kaggle quota reset Sat 00:00 UTC. ~2 concurrent GPU sessions per account, shared.

## New machine: recovery after a power cut (no UPS; live state in `STATUS_NEW_MACHINE.md` on branch `status-new-machine`, pushed every 20 min)
A cut mid-write can leave a truncated parquet that a rerun would reuse, so step 2 deletes unreadable ones. After a reboot:
1. `cd ~/Amizon && git pull`. Restart the status pusher: `setsid nohup experiments/status_push.sh > logs/status_push.log 2>&1 < /dev/null &`
2. `rm -f work/*.tmp && .venv/bin/python experiments/drop_bad_parquet.py` (deletes truncated parquets).
   Test caches must use the fr-norm prep (merged `a048dc3`, 26 Sep ~03:20 UTC). If `work/test_s1`
   is older than that, `rm -rf work/test_s1 work/test_cand work/test_*p1500_ap_ak*`. Train caches keep the old prep (≤0.1% differ).
3. Rerun Track D pairs; it skips whatever finished (train pairs reused if present):
   `setsid nohup .venv/bin/python code/business_entity_resolution/src/run.py pairs --n-train 2206821 --k-use 20 --pool 1500 --addr-partial --addr-keys --exclude-s1 work/ce/ce_train.parquet > logs/pairs_C_resume.log 2>&1 < /dev/null &`
   (`work/ce/ce_train.parquet` = the CE's 99,948 training S1, from Kaggle dataset `kaggle-A/amizon-ce-data`.)
4. Then: GPU scores (`experiments/kaggle_w.sh`, only pairs not already scored), `run.py train/predict` with `--emb-dir`.
   Big outputs go to Kaggle datasets (`-w`) as soon as written, so they survive a power cut.
5. v8 GPU scores (26 Sep): kernels `kaggle-B/amizon-{ce,embed}-{train,test}-w` score only the pairs the laptop hadn't
   (35.1M train, 15.8M test; datasets `amizon-miss-{train,test}-w`). They keep running on Kaggle through a local power cut.
   Fetch with `KAGGLE_API_TOKEN=<teammate token, ask the user; never stored> experiments/kaggle_w.sh get <kernel> work/gpu_new`,
   then concat with the laptop's scores (`work/prev_gpu/`: amizon-ce/amizon-embed train, amizon-ce-fast/amizon-embed test) into `work/gpu`.

## Track D request (user, 26 Sep 09:05): save v8's test features
Run v8's predict with **`--save-test-features`** (commit d0004cf): writes `work/test_features/part-*.parquet`
(features + i, j, prob). Needed for the planned pseudo-labelling add-on (retrain with confident French test pairs;
settings from `experiments/pseudo_sim.py`, running on the laptop). No other change to v8.

## Reminder (user asked, 26 Sep 10:05): cross-encoder pseudo-labelling — do AFTER ce2 results
Idea: fine-tune the best cross-encoder (ce2) further on confident FRENCH test pairs (pseudo-labels), then re-score
French test pairs. Unlike LightGBM pseudo-labelling (no gain in the US→India sim), the cross-encoder reads words, so
even easy French pairs can teach French vocabulary (Rue, SARL, Ste = Sainte). Allowed by the organisers.
Validate first with the same trick: fine-tune on confident India pairs, measure on India's real labels.

## Space rule (user, 26 Sep 10:08)
Anything that needs a lot of disk or RAM (full-data pairs, feature dumps, v8 train/predict, cross-encoder
pseudo-labelling data) runs on the **workstation**. The laptop only runs Kaggle jobs, small A/B tests and the docs/zip.

## Plan after v8 (agreed with the user, 26 Sep 11:40) — deadline 27 Sep 23:59; uploads: 4 left on 26th, 5 on 27th
Every change is judged by CV on the same setup; the FINAL pick is by CV (private LB decides), not by public LB.
| # | Item | Expected | Where | Notes |
|---|---|---|---|---|
| 1 | ce2 (e5-base OOF cross-encoder) as extra feature → v9 | +0.002–0.006 | Kaggle + laptop A/B | done: **+0.0005 only** (stack A/B 0.96103 vs control 0.96049), not adopted |
| 1b | **ce2 uncertain-pairs add-on for v8** (after v8's CV + test predictions exist) | ~+0.0005 | Kaggle (laptop) + workstation | models already trained, no retraining: Kaggle outputs of `kaggle-A/amizon-ce2-h0/h1` (`ce_model/`). Steps: (a) workstation exports v8's uncertain pairs, v8 OOF / test prob in [0.02, 0.98]: train rows `i, j` with `ce_half(i)`, test rows `i, j`; (b) score-only kernels (`ce.py --model-dir`): model h0 scores train half 1 + all test band, h1 scores train half 0 + all test band; ~1.5M+1.5M pairs, ~30–40 min on 2×T4 (1,650 pairs/s measured); test ce2 = mean of h0, h1; (c) 2nd-stage LightGBM on band pairs with [logit(v8 prob), ce2], GroupKFold 5 by S1; keep only if CV beats the control [logit(v8 prob)]; (d) replace band probs, redo threshold + one-owner rule, write the submission. `candidate_pairs.tsv` unchanged. |
| 2 | ~~Cross-encoder pseudo-labelling for France~~ | — | Kaggle | **REJECTED (26 Sep 14:35)**: India check (India = unseen country), 296k uncertain pairs: US-only CE AUC 0.8798, India F0.5 0.87480 (vs 0.82220 without CE) → **+ India pseudo-labels (.95/.05): AUC 0.8105, F0.5 0.83682 (−0.038)**. Pseudo-labels are the easy pairs and the CE drifts toward LightGBM's view. Upside: a CE that never saw a country still adds +0.053 there, so ce transfers to France as is. Kernel 1: pseudo .98/.02 0.83621; same steps US-only (control) 0.87244 → the pseudo-labels themselves cost −0.036 |
| N2 | **Error breakdown** (done 15:40 IST, `experiments/n1_n2.py`, `work/n1_n2_v7.log`) | — | laptop | v7: actual 0.96034, ceiling on its candidates 0.97144 → **never-found 0.0286**, FP-free 0.96403, FN-free 0.96776; US 0.974/0.986, India 0.940/0.949. v8 misses 262,880 true pairs (3.44%; India 4.9%, US 2.5%; `amizon-v8-missed`): **52% have an EMPTY candidate address** (US 66%), 27% of India's are native-script names |
| N1 | ~~2nd-stage list-context model~~ (rank/gap/counts per S1 and per candidate) | — | laptop | **rejected**: 0.95960 vs control [p] 0.96005 (v7 0.96034) |
| N5 | ~~Empty-address same-name block~~ | — | laptop | **rejected on recall/cost**: recovers 13.5–16.4% of v8 misses but precision 1.6–2.6% (+0.6–1.3 pairs/S1): same name, no address = unresolvable |
| N6 | ~~Voice-neutral skeleton (g≈k, b≈p, d≈t) for Indic names~~ | — | laptop | same-key share of India native-script misses 0.107 → 0.116 only |
| N4 | **Embedding kNN candidates** (e5-small, same country, top-50) | +0.003–0.006 if it works | Kaggle `amizon-n4` → workstation | **recall test running** (~16:30 IST) on v7's 522,674 misses (covers 192,003 of v8's). **Build overnight (~8–10 h) only if it recovers ≥25–30% of the non-empty-address misses**; else run #3 overnight |
| 3 | **Legal-word collision fix + dotted L.L.C./P.C. (prep.py, commit above)** | ~1% of names had a real word dropped (smith, summit, samiti, israel, place) | workstation | needs cache + pairs rebuild → include in the FINAL build if time allows (~4–5 h) |
| 4 | Candidate budget with blocking C: k = 10 / 15 vs 20 | ~0 F0.5, smaller file | workstation | organisers rank smaller candidate sets higher |
| 5 | Re-tune LightGBM (Optuna) on the new feature set | 0–0.002 | workstation | |
| 6 | France robustness: country-scale features, leave-one-country-out | small | workstation | |
| 7 | Seed averaging (3 LightGBM seeds) | 0–0.001 | workstation | |
| 8 | ~~Threshold probing on the public LB~~ | — | — | **done**: v8 @0.76 LB 0.974 = @0.72 → LB flat; keep the CV-optimal threshold, don't spend the 2nd probe |
Must-do on 27th: final pick by CV → docs (2 pages, blocking C, fr-norm, data-use disclosure incl. pseudo-labelling ruling) →
README flags (--pool 1500 --addr-partial --addr-keys --k-use 20 --exclude-s1 --params) → docstrings → zip from tag →
final upload by ~21:00.

## Packaging rule for v8 (track E)
The `v8` tag / zip must contain **prep.py from a048dc3** (fr-norm, before the legal-word fix 472a35c), because v8's
test caches were built with it (train caches: pre-fr-norm prep, ≤0.1% different, which the v8 docs note). When v8 is
final: commit a tree whose prep.py = `git show a048dc3:code/business_entity_resolution/src/prep.py`, tag it `v8`,
`./make_zip.sh output_v8 v8`. The legal-word fix (472a35c) goes into a final rebuild only after its own CV check
(rebuild train caches with it + `--block-sample` 400k CV vs the same without it).

## Queued after v8 (new machine): CV check of prep fix 472a35c
Baseline = ab7 arm C (`logs/ab7_addr.log`: CV 0.9669, recall 96.59%, pre-fr-norm prep). New arm must match it exactly,
from a worktree at 472a35c with its own work dir (never main `work/`, never in parallel with v8 jobs):
`run.py train --work work/prep_ab/new --n-train 400000 --block-sample --k-use 20 --pool 1500 --addr-partial --addr-keys --params code/business_entity_resolution/best_params.json`
No `--exclude-s1` (changes the seed-0 sample), no `--emb-dir`. If |Δ| ≤ 0.001 (≈ LightGBM run noise), add an a048dc3 arm too.
