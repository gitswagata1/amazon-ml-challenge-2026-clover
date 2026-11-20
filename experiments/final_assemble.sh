#!/usr/bin/env bash
# Final submission from a second-stage test-probability file: US/India at the CV threshold (+ conflict rule),
# France at the CV threshold (+ optional offset), then the two LB/simulation-backed France rejection rules, then validation.
#
#   experiments/final_assemble.sh WORK PROB JSON OUT [FR_OFFSET=0]
#   e.g. G1=work10/g_test_common_w9prep.parquet G2=work10/g2_test_diffaddr_sym_w9prep.parquet \
#        experiments/final_assemble.sh work10 work10/test_prob_v10ce2.parquet output_v10ce2/ce2_band.json output_final_v10 0
#
# WORK: work dir with test_s1/test_cand/emb_in (pairs universe for the France rules); PROB: i, j, prob for ALL scored
# test pairs (ce2_band.py --save-test-prob); JSON: the stage's threshold/exclusive choice (ce2_band.json).
# T (env, optional): US/India threshold overriding the json's (v12: 0.58 = CV optimum WITH the conflict rule,
# experiments/country_thr_cv.py; France = T + FR_OFFSET). G1/G2 (env): the P1 / P2' rule lists to apply (default: built into WORK if missing). The final uses lists built with
# work9's prep over v10's pairs (france_rule.py --work work9 --pairs <v10 prob>), the prep the P1 probe was run on.
set -euo pipefail
cd "$(dirname "$0")/.."
W=$1 PROB=$2 JSON=$3 OUT=$4 FR_OFFSET=${5:-0}   # France threshold offset: 0 with P5 (LB-validated); +0.04 was the pre-P5 setting
G1=${G1:-$W/g_test_common.parquet} G2=${G2:-$W/g2_test_diffaddr_sym.parquet}
PY=.venv/bin/python
PAIRS=$W/emb_in/test_pairs.parquet; [ -f "$PAIRS" ] || PAIRS=$PROB
$PY experiments/country_threshold.py --work "$W" --prob "$PROB" --json "$JSON" ${T:+--t "$T"} --country France --offset "$FR_OFFSET" \
    --conflict-delta 0.2 --conflict-countries US,India --out "${OUT}_thr"
[ -f "$G1" ] || $PY experiments/france_rule.py --work "$W" --pairs "$PAIRS" --out "$G1"
[ -f "$G2" ] || $PY experiments/france_rule.py --work "$W" --pairs "$PAIRS" --address different --out "$G2"
$PY experiments/france_probe.py --base "${OUT}_thr" --groups "$G1" --mode reject --out "${OUT}_p1"
$PY experiments/france_probe.py --base "${OUT}_p1" --groups "$G2" --mode reject --out "$OUT"
python3 utils/validate_submission.py -m "$OUT/matching_results.tsv" -c "$OUT/candidate_pairs.tsv" -t dataset/test --check-ids
