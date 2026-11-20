"""Decision layer: one-to-one assignment, per-country thresholds, the assembly of one build, and the country merge.

Subcommands (python pipeline/decide.py <subcommand> --help):

  thresholds
    Each Source 2/3 record goes to its most probable Source 1 record; pairs above the threshold are accepted (one country
    may use threshold + offset), with optional ambiguity rejection (drop a match when the runner-up scores >= 0.5 within
    delta). Writes matching_results.tsv and candidate_pairs.tsv (exactly the scored pairs).

  assemble
    thresholds + French look-alike rules + validator for one build:
        python pipeline/decide.py assemble WORK PROB JSON OUT [FR_OFFSET] [--t 0.58] [--g1 LIST --g2 LIST]

  merge
    Merges two submissions by country: rows of --countries from --take, all others from --base.
        python pipeline/decide.py merge --base <dir> --take <dir> --countries France --out <dir>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from run import write  # noqa: E402


# ── thresholds ────────────────────────────────────────────────────────────────
def main_thresholds():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--prob", type=Path, required=True, help="parquet i, j, prob for ALL test pairs")
    ap.add_argument("--json", type=Path, help="ce2_band.json / report json with threshold (+ exclusive)")
    ap.add_argument("--t", type=float, help="override the json threshold")
    ap.add_argument("--country", default="France")
    ap.add_argument("--offset", type=float, default=0.04, help="country threshold = t + offset")
    ap.add_argument("--conflict-delta", type=float, help="conflict rule: drop a record's winner when its 2nd-best S1 has "
                    "prob >= 0.5 and the gap is < delta (near-tie between look-alike S1; v9+ce2 CV +0.00018 @0.2)")
    ap.add_argument("--conflict-countries", default="US,India", help="countries the conflict rule applies to")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    js = json.loads(a.json.read_text()) if a.json else {}
    t = a.t if a.t is not None else js["threshold"]
    exclusive = js.get("exclusive", True)
    tc = round(t + a.offset, 4)
    s1p, cp = str(a.work / "test_s1" / "*.parquet"), str(a.work / "test_cand" / "*.parquet")
    c = pl.scan_parquet(s1p).select(i="idx", c="country").collect()
    allp = pl.read_parquet(a.prob, columns=["i", "j", "prob"])
    assert allp.height == allp.unique(["i", "j"]).height, "duplicate pairs in --prob"
    assert allp["prob"].null_count() == 0, "pairs without probability"
    p = allp.join(c, on="i")
    if a.conflict_delta is not None:  # 2nd-best prob of each record, before the one-owner filter
        second = (p.sort("prob", descending=True).with_columns(rk=pl.int_range(pl.len()).over("j"))
                  .filter(pl.col("rk") == 1).select("j", p2="prob"))
        p = p.join(second, on="j", how="left").with_columns(pl.col("p2").fill_null(0.0))
    if exclusive:  # one-owner rule
        p = p.filter(pl.col("prob") == pl.col("prob").max().over("j"))
    keep = pl.col("prob") >= pl.when(pl.col("c") == a.country).then(tc).otherwise(t)
    if a.conflict_delta is not None:
        near_tie = ((pl.col("p2") >= 0.5) & (pl.col("prob") - pl.col("p2") < a.conflict_delta)
                    & pl.col("c").is_in(a.conflict_countries.split(",")))
        print(f"conflict rule (delta {a.conflict_delta}, {a.conflict_countries}): "
              f"{p.filter(keep & near_tie).height} accepted winners dropped", flush=True)
        keep = keep & ~near_tie
    q = p.filter(keep).select("i", "j")
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "matching_results.tsv", "matched_entity_ids", s1p, cp, q)
    write(a.out / "candidate_pairs.tsv", "candidate_entity_ids", s1p, cp, allp.select("i", "j"))
    n = q.join(c, on="i").group_by("c").len().sort("c").rows()
    print(f"wrote {a.out}: {q.height} matches {n} (t={t}, exclusive={exclusive}, {a.country} t={tc}); "
          f"candidates = {allp.height} scored pairs", flush=True)


# ── merge ────────────────────────────────────────────────────────────────────
def main_merge():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--take", type=Path, required=True)
    ap.add_argument("--countries", default="France")
    ap.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False)  # noqa: E731
    ids = (rd(a.test_dir / "test_source1.tsv").filter(pl.col("country").is_in(a.countries.split(",")))
           .select(source1_entity_id="entity_id"))
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        b, t = rd(a.base / f), rd(a.take / f)
        col = b.columns[1]
        assert b.columns == t.columns and b.height == t.height, f"{f}: files do not align"
        take = ids["source1_entity_id"].implode()
        m = (b.join(t.rename({col: "t"}), on="source1_entity_id", how="left", maintain_order="left")
             .with_columns(pl.when(pl.col("source1_entity_id").is_in(take)).then(pl.col("t")).otherwise(pl.col(col)).alias(col))
             .select(b.columns))
        d = m.join(b, on="source1_entity_id", suffix="_b").filter(pl.col(col).fill_null("") != pl.col(col + "_b").fill_null(""))
        assert d.join(ids, on="source1_entity_id", how="anti").height == 0, "a row outside --countries changed"
        print(f"{f}: {d.height} rows from {a.take} ({a.countries})", flush=True)
        m.write_csv(a.out / f, separator="\t", quote_style="never", null_value="")
    subprocess.run([sys.executable, "utils/validate_submission.py", "-m", str(a.out / "matching_results.tsv"),
                    "-c", str(a.out / "candidate_pairs.tsv"), "-t", str(a.test_dir), "--check-ids"], check=True)


# ── assemble ───────────────────────────────────────────────────────────────
def main_assemble():
    """Decision layer for one build, then validation: per-country thresholds with ambiguity rejection for US/India,
    French look-alike rules (lists built into WORK when missing), validator. Run from the package root."""
    ap = argparse.ArgumentParser()
    ap.add_argument("work", help="work folder of the build (test_s1 / test_cand parquet)")
    ap.add_argument("prob", help="i, j, prob for every scored test pair (rerank.py stack --save-test-prob)")
    ap.add_argument("json", help="the stage's threshold / one-owner choice (ce2_band.json)")
    ap.add_argument("out")
    ap.add_argument("fr_offset", nargs="?", default="0", help="France threshold = US/India threshold + offset")
    ap.add_argument("--t", help="US/India threshold (overrides the json)")
    ap.add_argument("--g1", help="same-address look-alike pair list (default <work>/g_test_common.parquet)")
    ap.add_argument("--g2", help="different-address look-alike pair list (default <work>/g2_test_diffaddr_sym.parquet)")
    a = ap.parse_args()
    france = str(Path(__file__).resolve().with_name("france.py"))
    g1 = a.g1 or f"{a.work}/g_test_common.parquet"
    g2 = a.g2 or f"{a.work}/g2_test_diffaddr_sym.parquet"
    run = lambda *args: subprocess.run([sys.executable, *args], check=True)  # noqa: E731
    run(__file__, "thresholds", "--work", a.work, "--prob", a.prob, "--json", a.json, *(["--t", a.t] if a.t else []),
        "--country", "France", "--offset", a.fr_offset, "--conflict-delta", "0.2", "--conflict-countries", "US,India",
        "--out", f"{a.out}_thr")
    if not Path(g1).exists():
        run(france, "rules", "--work", a.work, "--pairs", a.prob, "--out", g1)
    if not Path(g2).exists():
        run(france, "rules", "--work", a.work, "--pairs", a.prob, "--address", "different", "--out", g2)
    run(france, "apply", "--base", f"{a.out}_thr", "--groups", g1, "--mode", "reject", "--out", f"{a.out}_p1")
    run(france, "apply", "--base", f"{a.out}_p1", "--groups", g2, "--mode", "reject", "--out", a.out)
    run("utils/validate_submission.py", "-m", f"{a.out}/matching_results.tsv", "-c", f"{a.out}/candidate_pairs.tsv",
        "-t", "dataset/test", "--check-ids")


SUBCOMMANDS = {"thresholds": main_thresholds, "assemble": main_assemble, "merge": main_merge}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        sys.exit("usage: python decide.py {" + " | ".join(SUBCOMMANDS) + "} [options]")
    SUBCOMMANDS[sys.argv.pop(1)]()
