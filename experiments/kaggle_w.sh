#!/usr/bin/env bash
# GPU scoring on Kaggle for the new machine's pairs (Track D), under whichever account the CLI is logged in as
# (e.g. a teammate's: KAGGLE_API_TOKEN=... in the environment only, never written to disk). All datasets private.
#
#   experiments/kaggle_w.sh data amizon-text-w     work/text                       # raw text (idx = TSV row order)
#   experiments/kaggle_w.sh data amizon-cemodel-w  work/prev_gpu/amizon-ce/ce_model   # trained e5-small cross-encoder
#   experiments/kaggle_w.sh data amizon-miss-train-w work/miss_train                 # {split}_pairs.parquet to score
#   experiments/kaggle_w.sh embed train amizon-miss-train-w    # kernel amizon-embed-train-w -> train_emb.parquet
#   experiments/kaggle_w.sh ce    train amizon-miss-train-w    # kernel amizon-ce-train-w (2xT4) -> train_ce.parquet
#   experiments/kaggle_w.sh status amizon-ce-train-w
#   experiments/kaggle_w.sh get    amizon-ce-train-w work/gpu_new
set -euo pipefail
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
U=$($K config view 2>/dev/null | sed -n 's/^- username: //p')
[ -n "$U" ] || { echo "kaggle CLI not authenticated"; exit 1; }
S=code/business_entity_resolution/src

kernel() {  # $1 kernel slug, $2 source .py, $3 entry-point args, $4 json dataset list
    local kd=work/kg/$1; mkdir -p "$kd"
    sed '/^if __name__ == "__main__":/,$d' "$2" > "$kd/$1.py"
    cat >> "$kd/$1.py" <<EOF
if __name__ == "__main__":
    import glob, os, sys
    find = lambda pat: os.path.dirname(sorted(glob.glob("/kaggle/input/**/" + pat, recursive=True))[0])
    sys.argv = ["$(basename "$2")", "--text", find("train_s1.parquet"), "--pairs", find("*_pairs.parquet"),
                "--out", "/kaggle/working"$3]
    main()
EOF
    cat > "$kd/kernel-metadata.json" <<EOF
{"id": "$U/$1", "title": "$1", "code_file": "$1.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": true, "dataset_sources": $4}
EOF
}

case "${1:-}" in
data)
    d=work/kg/$2; rm -rf "$d"; mkdir -p "$d"; cp -l "$3"/* "$d"/
    echo "{\"title\": \"$2\", \"id\": \"$U/$2\", \"licenses\": [{\"name\": \"other\"}]}" > "$d/dataset-metadata.json"
    if $K datasets status "$U/$2" >/dev/null 2>&1; then $K datasets version -p "$d" -m "update $(date -u +%FT%H:%M)"
    else $K datasets create -p "$d"; fi ;;
embed)
    kernel "amizon-embed-$2-w" $S/embed.py "" "[\"$U/amizon-text-w\", \"$U/$3\"]"
    $K kernels push -p "work/kg/amizon-embed-$2-w" --accelerator NvidiaTeslaT4 ;;
ce)
    kernel "amizon-ce-$2-w" $S/ce.py ', "--model-dir", find("config.json")' \
        "[\"$U/amizon-text-w\", \"$U/amizon-cemodel-w\", \"$U/$3\"]"
    $K kernels push -p "work/kg/amizon-ce-$2-w" --accelerator NvidiaTeslaT4 ;;
status) $K kernels status "$U/$2" ;;
get) mkdir -p "$3"; $K kernels output "$U/$2" -p "$3" --file-pattern '_(emb|ce)\.parquet$'; ls -la "$3" ;;
*) sed -n 2,13p "$0"; exit 1 ;;
esac
