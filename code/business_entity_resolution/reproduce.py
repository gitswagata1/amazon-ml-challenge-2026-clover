"""Rebuild output/matching_results.tsv and output/candidate_pairs.tsv (public LB 0.986691) from the competition data.

    pip install -r code/business_entity_resolution/requirements.txt
    pip install -r code/business_entity_resolution/requirements-gpu.txt      # torch + transformers (CUDA)
    python code/business_entity_resolution/reproduce.py --data /path/to/student_resource

--data is the student_resource folder holding dataset/ and utils/. Every stage except `final` needs a CUDA GPU.
A step whose output already exists is skipped, so an interrupted run resumes. `--stage final` (repeatable) runs
only the named stages. Linux, macOS or WSL.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # package root: the folder that holds code/
PKG = "code/business_entity_resolution"
S, P = f"{PKG}/src", f"{PKG}/src/pipeline"
PARAMS = f"{PKG}/best_params.json"

FLAGS9 = ["--work", "work9", "--n-train", "2206821", "--k-use", "20", "--pool", "1500", "--addr-partial", "--addr-keys",
          "--rescue", "--knn", "work/n4", "--knn-top", "5", "--rare-empty", "--exclude-s1", "work/ce/ce_train.parquet"]
FLAGS10 = ["--work", "work10", "--n-train", "2206821", "--k-use", "20", "--pool", "1500", "--addr-partial", "--addr-keys",
           "--rescue", "--knn", "work/n4", "--knn-top", "5", "--knn2", "work/ret", "--knn2-top", "10", "--rare-empty",
           "--exclude-s1", "work/ce/ce_train.parquet", "--tfidf-all"]
TAG9, TAG10 = "k30_p1500_ap_ak_use20_rs_knn5_re", "k30_p1500_ap_ak_use20_rs_knn5_rk10_re"


def export_text(work):
    """Raw name/address per record for the GPU scripts: <work>/text/{split}_{s1,cand}.parquet."""
    import polars as pl
    (Path(work) / "text").mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for w in ("s1", "cand"):
            (pl.scan_parquet(f"{work}/{split}_{w}/*.parquet").select("idx", "business_name", "business_address")
             .sink_parquet(f"{work}/text/{split}_{w}.parquet"))


def gpu_features(work):
    """e5 cosines + small cross-encoder score for every candidate pair of a build."""
    return [
        (f"{work}/text/test_cand.parquet", lambda: export_text(work)),
        (f"{work}/gpu/test_emb.parquet", [f"{S}/embed.py", "--text", f"{work}/text", "--pairs", f"{work}/emb_in",
                                          "--out", f"{work}/gpu"]),
        (f"{work}/gpu/test_ce.parquet", [f"{S}/ce.py", "--text", f"{work}/text", "--pairs", f"{work}/emb_in",
                                         "--out", f"{work}/gpu", "--model-dir", "work/gpu/ce_model"]),
    ]


def rerank_band(work):
    """Score a build's uncertainty band with both re-ranker halves."""
    return [(f"{work}/ce2band/h{h}/test_ce.parquet",
             [f"{S}/ce.py", "--text", f"{work}/text", "--pairs", f"{work}/band/h{h}", "--out", f"{work}/ce2band/h{h}",
              "--model-dir", f"work/ce2_h{h}/ce_model"]) for h in (0, 1)]


def stages(a):
    return {
        # normalised records + raw text + small cross-encoder
        "base": [
            ("work/train_pairs_k30_p1500_ap_ak_use20.parquet",
             [f"{S}/run.py", "pairs", "--work", "work", "--n-train", "2206821", "--k-use", "20", "--pool", "1500",
              "--addr-partial", "--addr-keys"]),
            ("work/text/test_cand.parquet", lambda: export_text("work")),
            ("work/ce/ce_train.parquet", [f"{S}/ce_data.py", "--n-train", "600000", "--out", "work/ce"]),
            ("work/gpu/ce_model", [f"{S}/ce.py", "--text", "work/text", "--pairs", "work/ce", "--out", "work/gpu"]),
        ],
        # two nearest-neighbour candidate sources: plain e5 and the fine-tuned dense retriever
        "neighbours": [
            ("work/retriever", [f"{P}/candidates.py", "train-retriever", "--work", "work", "--ce", "work/ce/ce_train.parquet",
                                "--out", "work/retriever"]),
            *[(f"work/{d}/{s}_knn.parquet", [f"{P}/candidates.py", "search", "--work", "work", "--split", s,
                                             "--model", m, "--out", f"work/{d}"])
              for s in ("train", "test")
              for d, m in (("n4", "intfloat/multilingual-e5-small"), ("ret", "work/retriever"))],
        ],
        # build v9 (source of the French rows): candidates, features, classifier
        "v9": [
            ("work9/test_cand", [f"{S}/run.py", "pairs", *FLAGS9]),
            ("work9/tfw_test.parquet", [f"{P}/extra_features.py", "word-tfidf", "--work", "work9", "--tag", TAG9]),
            ("work9/rarity_test.parquet", [f"{P}/extra_features.py", "rarity", "--work", "work9", "--tag", TAG9]),
            ("work9/train_features.parquet", [f"{S}/run.py", "train", *FLAGS9, "--features-only"]),
            ("work9/test_features", [f"{S}/run.py", "predict", *FLAGS9, "--features-only"]),
            *gpu_features("work9"),
            ("work9/test_prob_v9.parquet",
             [f"{P}/classifier.py", "--work", "work9", "--params", PARAMS,
              "--extra", "work9/gpu/{split}_emb.parquet", "--extra", "work9/gpu/{split}_ce.parquet",
              "--extra", "work9/tfw_{split}.parquet", "--extra", "work9/rarity_{split}.parquet",
              "--out", "output_v9", "--tag", "v9"]),
        ],
        # e5-base re-ranker (one model per S1 half), stacking on v9, French look-alike lists
        "rerank": [
            *[step for h in (0, 1) for step in (
                (f"work/ce2/ce_train_h{h}.parquet",
                 [f"{S}/ce_data.py", "--half", str(h), "--n-ce", "200000", "--k-use", "20", "--out", "work/ce2"]),
                (f"work/ce2_h{h}/ce_model",
                 [f"{S}/ce.py", "--text", "work/text", "--pairs", "work/ce2", "--train-file", f"ce_train_h{h}.parquet",
                  "--out", f"work/ce2_h{h}", "--model", "intfloat/multilingual-e5-base", "--lr", "3e-5",
                  "--epochs", "1", "--batch", "64"]))],
            ("work9/band/h1", [f"{P}/rerank.py", "export-band", "--oof", "work9/oof_v9.parquet",
                               "--test-prob", "work9/test_prob_v9.parquet", "--out", "work9/band"]),
            *rerank_band("work9"),
            ("work9/test_prob_v9ce2.parquet",
             [f"{P}/rerank.py", "stack", "--work", "work9", "--band", "work9/ce2band", "--oof", "work9/oof_v9.parquet",
              "--test-prob", "work9/test_prob_v9.parquet", "--meta", "work9/tune_meta.parquet",
              "--report", "work9/report_v9.json", "--src", "output_v9", "--out", "output_v9ce2"]),
            ("work9/g_test_common.parquet", [f"{P}/france.py", "rules", "--work", "work9", "--pairs",
                                             "work9/test_prob_v9.parquet", "--out", "work9/g_test_common.parquet"]),
            ("work9/g2_test_diffaddr_sym.parquet",
             [f"{P}/france.py", "rules", "--work", "work9", "--pairs", "work9/test_prob_v9.parquet",
              "--address", "different", "--out", "work9/g2_test_diffaddr_sym.parquet"]),
        ],
        # two domain-adapted French cross-encoders (self-training on the unlabelled French test records)
        "france": [
            ("work9/frce/pairs/ce_train.parquet", [f"{P}/france.py", "self-train-data", "--out", "work9/frce",
                                                   "--score", "work9/test_prob_v9ce2.parquet"]),
            *[(f"work9/frce_{m}/test_ce.parquet",
               [f"{S}/ce.py", "--text", "work9/frce/text", "--pairs", "work9/frce/pairs", "--out", f"work9/frce_{m}",
                "--model", f"work/ce2_h{h}/ce_model", "--lr", "2e-5", "--epochs", "1", "--batch", "64"])
              for m, h in (("A", 0), ("B", 1))],
        ],
        # build v10 (source of the US/India rows): retriever candidates, features, 3-seed classifier, re-ranker
        "v10": [
            ("work10/test_cand", [f"{S}/run.py", "pairs", *FLAGS10]),
            ("work10/rarity_test.parquet", [f"{P}/extra_features.py", "rarity", "--work", "work10", "--tag", TAG10]),
            ("work10/train_features.parquet", [f"{S}/run.py", "train", *FLAGS10, "--features-only"]),
            ("work10/test_features", [f"{S}/run.py", "predict", *FLAGS10, "--features-only"]),
            *gpu_features("work10"),
            ("work10/test_prob_v10.parquet",
             [f"{P}/classifier.py", "--work", "work10", "--params", PARAMS,
              "--extra", "work10/gpu/{split}_emb.parquet", "--extra", "work10/gpu/{split}_ce.parquet",
              "--extra", "work10/rarity_{split}.parquet", "--seeds", "3", "--out", "output_v10", "--tag", "v10"]),
            ("work10/band/h1", [f"{P}/rerank.py", "export-band", "--oof", "work10/oof_v10.parquet",
                                "--test-prob", "work10/test_prob_v10.parquet", "--out", "work10/band"]),
            *rerank_band("work10"),
            ("work10/test_prob_v10ce2.parquet",
             [f"{P}/rerank.py", "stack", "--work", "work10", "--band", "work10/ce2band", "--oof", "work10/oof_v10.parquet",
              "--test-prob", "work10/test_prob_v10.parquet", "--meta", "work10/tune_meta.parquet",
              "--src", "output_v10", "--out", "output_v10ce2"]),
        ],
        # per-country decisions, merge, French noise-variant recovery -> output/
        "final": [
            # US/India rows: threshold 0.58 + ambiguity rejection (this file's French rows are replaced below)
            ("output_final_usin",
             [f"{P}/decide.py", "assemble", "work10", "work10/test_prob_v10ce2.parquet", "output_v10ce2/ce2_band.json",
              "output_final_usin", "0.10", "--t", "0.58", "--g1", "work9/g_test_common.parquet",
              "--g2", "work9/g2_test_diffaddr_sym.parquet"]),
            # French rows: v9 stacking on S1 half 1 with French models A + B averaged; France threshold 0.72 + 0.19
            ("work9/test_prob_final_fr.parquet",
             [f"{P}/rerank.py", "stack", "--work", "work9", "--band", "work9/ce2band", "--oof", "work9/oof_v9.parquet",
              "--test-prob", "work9/test_prob_v9.parquet", "--meta", "work9/tune_meta.parquet", "--src", "output_v9",
              "--out", "output_v9_fr", "--save-test-prob", "work9/test_prob_final_fr.parquet", "--half", "1",
              "--margin", "-1", "--frce", a.french_a, a.french_b, "--frce-min-cov", "0.95"]),
            ("output_final_fr", [f"{P}/decide.py", "assemble", "work9", "work9/test_prob_final_fr.parquet",
                                 "output_v9_fr/ce2_band.json", "output_final_fr", "0.19"]),
            ("output_merged", [f"{P}/decide.py", "merge", "--base", "output_final_usin", "--take", "output_final_fr",
                               "--countries", "France", "--out", "output_merged"]),
            (None, [f"{P}/france.py", "recover", "--base", "output_merged", "--reverse-rules", "--out", "output"]),
        ],
    }


def run_step(out, cmd):
    if out and Path(out).exists():
        print(f"· skip   {out}", flush=True)
        return
    print(f"▶ build  {out or 'output'}", flush=True)
    if callable(cmd):
        cmd()
    else:
        subprocess.run([sys.executable, *cmd], check=True)


def link_data(data):
    """dataset/ and utils/ next to code/ (the scripts read these relative paths)."""
    for name in ("dataset", "utils"):
        target = (Path(data) / name).resolve()
        if not target.is_dir():
            sys.exit(f"error: {data} has no {name}/ folder (expected the student_resource folder)")
        link = ROOT / name
        if link.is_symlink() or not link.exists():
            if link.is_symlink():
                link.unlink()
            link.symlink_to(target)


def check(names):
    missing = []
    for mod in ("polars", "lightgbm", "rapidfuzz", "sklearn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(f"error: missing packages {missing}. Run: pip install -r {PKG}/requirements.txt")
    if any(n != "final" for n in names):
        try:
            import torch
            assert torch.cuda.is_available()
        except (ImportError, AssertionError):
            sys.exit(f"error: stages {[n for n in names if n != 'final']} need a CUDA GPU with torch installed "
                     f"(pip install -r {PKG}/requirements-gpu.txt)")
    try:
        gb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) / 2**20
        if gb < 48:
            print(f"warning: {gb:.0f} GB RAM; candidate generation peaks at about 36 GB", flush=True)
    except (OSError, StopIteration):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="the student_resource folder (with dataset/ and utils/); "
                                   "optional if both already sit next to code/")
    ap.add_argument("--stage", action="append", choices=("base", "neighbours", "v9", "rerank", "france", "v10", "final"),
                    help="run only this stage (repeatable); default: all, in order")
    ap.add_argument("--french-a", default="work9/frce_A", help=argparse.SUPPRESS)   # French model A scores folder
    ap.add_argument("--french-b", default="work9/frce_B", help=argparse.SUPPRESS)   # French model B scores folder
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.data:
        link_data(a.data)
    if not Path("dataset/test/test_source1.tsv").exists() or not Path("utils/validate_submission.py").exists():
        sys.exit("error: dataset/ or utils/ not found. Pass: --data /path/to/student_resource")
    all_stages = stages(a)
    names = a.stage or list(all_stages)
    check(names)
    for name in names:
        print(f"══ stage {name}", flush=True)
        for out, cmd in all_stages[name]:
            run_step(out, cmd)
    print("done: output/matching_results.tsv, output/candidate_pairs.tsv", flush=True)


if __name__ == "__main__":
    main()
