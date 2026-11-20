"""Cross-encoder re-ranking of uncertain pairs and the stacked meta-classifier.

Subcommands (python pipeline/rerank.py <subcommand> --help):

  export-band
    Writes the pairs in the uncertainty band (classifier probability 0.02-0.98) for the multilingual-e5-base
    cross-encoder: for each of the two cross-encoder models, the training pairs of the other half of the Source 1 records
    (so every training pair gets an out-of-fold score), and all test band pairs.
        python pipeline/rerank.py export-band --oof work9/oof_v9.parquet --test-prob work9/test_prob_v9.parquet --out work9/band

  stack
    Stacked LightGBM meta-model on [logit(classifier probability), cross-encoder logit] over the band pairs, 5-fold CV by
    Source 1 record, compared with a control model that sees only the classifier probability. Writes the re-scored test
    probabilities and the chosen threshold. With --frce, French pairs use the French cross-encoders' scores in place of
    the re-ranker score (averaged over the given folders); --half fits and scores on one half of the Source 1 records.
        python pipeline/rerank.py stack --work work10 --band work10/ce2band --oof work10/oof_v10.parquet \
            --test-prob work10/test_prob_v10.parquet --meta work10/tune_meta.parquet --src output_v10 --out output_v10ce2
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import polars as pl
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))  # git repo
from ce_data import ce_half  # noqa: E402
from run import decide, write  # noqa: E402


SECOND = dict(n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=100, verbose=-1)
GRID = np.arange(0.2, 0.96, 0.02)


# ── export-band ─────────────────────────────────────────────────────────────
def main_export_band():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, required=True, help="i, j, y, oof")
    ap.add_argument("--test-prob", type=Path, required=True, help="i, j, prob")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    band = lambda f, c: pl.read_parquet(f, columns=["i", "j", c]).filter(pl.col(c).is_between(0.02, 0.98)).select("i", "j")
    tr, te = band(a.oof, "oof"), band(a.test_prob, "prob")
    for h in (0, 1):
        d = a.out / f"h{h}"
        d.mkdir(parents=True, exist_ok=True)
        tr.filter(ce_half(pl.col("i")) == 1 - h).write_parquet(d / "train_pairs.parquet")
        te.write_parquet(d / "test_pairs.parquet")
        print(f"h{h}: train {tr.filter(ce_half(pl.col('i')) == 1 - h).height}, test {te.height}")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def f05(o, meta, t, exclusive):
    """Macro F0.5 over all sampled S1. o: i, j, y, oof."""
    q = o.filter(pl.col("oof") == pl.col("oof").max().over("j")) if exclusive else o
    q = q.filter(pl.col("oof") >= t)
    per = (meta.join(q.group_by("i").agg(npred=pl.len(), tp=pl.col("y").sum()), on="i", how="left")
           .fill_null(0).with_columns(pl.col("tp").cast(pl.Float64)))
    P, R = pl.col("tp") / pl.col("npred"), pl.col("tp") / pl.col("ntrue")
    return per.select(pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0)
                      .when(pl.col("tp") == 0).then(0.0)
                      .otherwise(1.25 * P * R / (0.25 * P + R)).mean()).item()


# ── stack ──────────────────────────────────────────────────────────────────
def main_stack():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--band", type=Path, default=Path("work/ce2band"), help="h0/, h1/ with train_ce + test_ce.parquet")
    ap.add_argument("--out", type=Path, default=Path("output_v81ce2"))
    ap.add_argument("--src", type=Path, default=Path("output_v81"), help="base submission: candidate_pairs.tsv copied")
    ap.add_argument("--oof", type=Path, help="base OOF (i, j, y, oof); default <work>/oof_v81.parquet")
    ap.add_argument("--test-prob", type=Path, help="base test probs (i, j, prob); default <work>/test_prob_v81.parquet")
    ap.add_argument("--meta", type=Path, help="all sampled S1 (i, ntrue); default <work>/tune_meta.parquet")
    ap.add_argument("--report", type=Path, help="base report json to compare the reproduced CV with (optional)")
    ap.add_argument("--ce3", type=Path, help="optional extra cross-encoder scores covering half 1 of the Source 1 "
                                                 "records: compares the model with and without them on that half "
                                                 "(not used by reproduce.py)")
    ap.add_argument("--margin", type=float, help="min CV gain over the control model needed to write output "
                                                  "(default: any gain; 0.0003 with --ce3)")
    ap.add_argument("--save-test-prob", type=Path, help="final test probs (i, j, prob); default "
                                                         "<work>/test_prob_<out name minus 'output_'>.parquet")
    ap.add_argument("--frce", type=Path, nargs="+", help="folder(s) with test_ce.parquet of the French cross-encoders: every "
                                                  "French test pair with base prob >= 0.02 is re-scored by the stage-2 "
                                                  "model with this score in place of the re-ranker score; several "
                                                  "folders -> stage-2 outputs averaged")
    ap.add_argument("--save-oof", type=Path, help="write the stage-2 out-of-fold predictions (i, j, y, oof)")
    ap.add_argument("--frce-min-cov", type=float, default=0.98, help="min share of French pairs (p >= 0.02) with a French cross-encoder score")
    ap.add_argument("--write-arm", choices=("new", "control"), default="new",
                    help="which model to fit and write: new = with the extra scores, control = without")
    ap.add_argument("--half", type=int, choices=(0, 1), help="fit and score the stage on the S1 records of this half only "
                    "(the final's French rows use half 1)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.oof, a.test_prob = a.oof or a.work / "oof_v81.parquet", a.test_prob or a.work / "test_prob_v81.parquet"
    a.meta = a.meta or a.work / "tune_meta.parquet"
    a.margin = a.margin if a.margin is not None else (0.0003 if a.ce3 else 1e-9)  # any gain > 0
    a.save_test_prob = a.save_test_prob or a.work / f"test_prob_{a.out.name.removeprefix('output_')}.parquet"
    ce = lambda f, name="ce2": pl.read_parquet(f).rename({"ce": name})  # noqa: E731
    half = ((pl.col("i").cast(pl.UInt64) * 2654435761) % 2**32) % 2  # ce_data.ce_half
    for h, other in (("h0", 1), ("h1", 0)):  # leak check: model h scored only the OTHER half's train pairs
        t = pl.read_parquet(a.band / h / "train_ce.parquet", columns=["i", "j"])
        halves = t.select(half.alias("h"))["h"].unique().to_list()
        assert halves == [other], f"{h} train scores cover halves {halves}, expected only [{other}]: leakage"
        assert t.height == t.select("i", "j").n_unique(), f"{h} train scores have duplicate pairs"
    tr_ce = pl.concat([ce(a.band / "h0/train_ce.parquet"), ce(a.band / "h1/train_ce.parquet")])
    te_ce = (ce(a.band / "h0/test_ce.parquet").join(ce(a.band / "h1/test_ce.parquet"), on=["i", "j"], suffix="_b")
             .select("i", "j", ce2=(pl.col("ce2") + pl.col("ce2_b")) / 2))
    o = pl.read_parquet(a.oof, columns=["i", "j", "y", "oof"])
    meta = pl.read_parquet(a.meta, columns=["i", "ntrue"])
    arms = [("B control [logit(oof)]", ["lo"]), ("A [logit(oof), ce2]", ["lo", "ce2"])]
    if a.ce3:  # the extra scores cover half 1 only: fit and score on that half
        c3 = ce(a.ce3 / "train_ce.parquet", "ce3")
        halves = c3.select(half.alias("h"))["h"].unique().to_list()
        assert halves == [1], f"ce3 train scores cover halves {halves}, expected only [1] (half-0 model): leakage"
        o, meta = o.filter(half == 1), meta.filter(half == 1)
        tr_ce = tr_ce.join(c3, on=["i", "j"], how="inner")
        te_ce = te_ce.join(ce(a.ce3 / "test_ce.parquet", "ce3"), on=["i", "j"], how="inner")
        arms = [("A2 [logit(oof), ce2]", ["lo", "ce2"]), ("A3 [logit(oof), ce2, ce3]", ["lo", "ce2", "ce3"])]
    if a.half is not None:  # fit and score on one half of the Source 1 records
        o, meta = o.filter(half == a.half), meta.filter(half == a.half)
    o = o.with_row_index("r")
    band = o.join(tr_ce, on=["i", "j"], how="inner").sort("r")
    band = band.with_columns(lo=pl.Series(logit(band["oof"].to_numpy())))
    print(f"[ce2_band] train pairs {o.height}, band {band.height} (y rate {band['y'].mean():.3f}), "
          f"test band {te_ce.height}, sampled S1 {meta.height}{' (half 1 only, ce3 mode)' if a.ce3 else ''}", flush=True)

    def score(prob_band=None):
        """Full decision: prob_band on band pairs, the base oof elsewhere; best over the grid x one-owner on/off."""
        oo = o.select("i", "j", "y", "oof") if prob_band is None else (
            o.join(band.select("r", p2=pl.Series(prob_band, dtype=pl.Float32)), on="r", how="left")
             .select("i", "j", "y", oof=pl.coalesce("p2", "oof")))
        return max((f05(oo, meta, t, ex), round(float(t), 2), ex) for ex in (True, False) for t in GRID)

    v81 = score()
    ref = ""
    if a.report and a.report.exists():
        rep = json.loads(a.report.read_text())
        ref = f" ({a.report.name}: {rep['cv_f05']:.5f} @ {rep['threshold']})"
    print(f"base reproduced ({a.oof.name}): CV {v81[0]:.5f} @ t={v81[1]} exclusive={v81[2]}{ref}", flush=True)
    y, g = band["y"].to_numpy(), band["i"].to_numpy()
    res, p2s = {}, {}
    for name, cols in arms:
        X = band.select(cols).to_numpy().astype(np.float32)
        p2 = np.zeros(len(y), np.float32)
        for tr, va in GroupKFold(5).split(X, y, g):
            p2[va] = lgb.LGBMClassifier(**SECOND, n_jobs=a.threads).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        res[name], p2s[name] = score(p2), p2
        print(f"{name}: CV {res[name][0]:.5f} @ t={res[name][1]} exclusive={res[name][2]}", flush=True)
    (cname, ccols), (nname, ncols) = arms
    C, N = res[cname], res[nname]
    gain = N[0] - C[0]
    print(f"{nname.split()[0]} - {cname.split()[0]} = {gain:+.5f}; vs base = {N[0] - v81[0]:+.5f} (margin {a.margin})", flush=True)
    if gain < a.margin:
        print("the cross-encoder model does not beat the control by the margin: nothing written", flush=True)
        return
    if a.write_arm == "control":  # write the control model, whatever the CV says
        nname, ncols, N = cname, ccols, C
    if a.save_oof:
        o.join(band.select("r", p2=pl.Series(p2s[nname], dtype=pl.Float32)), on="r", how="left") \
         .select("i", "j", "y", oof=pl.coalesce("p2", "oof")).write_parquet(a.save_oof)
        print(f"stage-2 OOF ({nname}) -> {a.save_oof}", flush=True)
    m = lgb.LGBMClassifier(**SECOND, n_jobs=a.threads).fit(band.select(ncols).to_numpy().astype(np.float32), y)
    tp = pl.read_parquet(a.test_prob, columns=["i", "j", "prob"])
    tb = tp.join(te_ce, on=["i", "j"], how="inner")
    n_band = tp.filter(pl.col("prob").is_between(0.02, 0.98)).height  # the exported band
    assert tb.height == n_band, f"test band {n_band} pairs but {tb.height} scored: truncated or mismatched CE output"
    tb = tb.with_columns(lo=pl.Series(logit(tb["prob"].to_numpy())))
    p2 = m.predict_proba(tb.select(ncols).to_numpy().astype(np.float32))[:, 1]
    tp = tp.join(tb.select("i", "j", p2=pl.Series(p2, dtype=pl.Float32)), on=["i", "j"], how="left", maintain_order="left") \
           .with_columns(prob=pl.coalesce("p2", "prob")).select("i", "j", "prob")
    if a.frce:  # French pairs: the French cross-encoders' scores replace the re-ranker score
        fr = pl.scan_parquet(str(a.work / "test_s1" / "*.parquet")).filter(pl.col("country") == "France").select(i="idx").collect()
        base = pl.read_parquet(a.test_prob, columns=["i", "j", "prob"]).join(fr, on="i").filter(pl.col("prob") >= 0.02)
        pfs = []
        for d in a.frce:
            x = base.join(ce(d / "test_ce.parquet"), on=["i", "j"], how="inner")
            assert x.height >= a.frce_min_cov * base.height, f"{d}: France-adapted scores cover {x.height} of {base.height} French pairs (p >= 0.02)"
            if x.height < base.height:
                print(f"{d}: {base.height - x.height} French pairs without a French cross-encoder score", flush=True)
            if "ce3" in ncols:  # extra scores exist for band pairs only; other French pairs get NaN (LightGBM's missing branch)
                x = x.join(te_ce.select("i", "j", "ce3"), on=["i", "j"], how="left")
            x = x.with_columns(lo=pl.Series(logit(x["prob"].to_numpy())))
            pfs.append(x.select("i", "j", pf=pl.Series(m.predict_proba(x.select(ncols).to_numpy().astype(np.float32))[:, 1], dtype=pl.Float32)))
        pf = pl.concat(pfs).group_by("i", "j").agg(pl.col("pf").mean())  # average of the models that scored the pair
        tp = tp.join(pf, on=["i", "j"], how="left", maintain_order="left") \
               .with_columns(prob=pl.coalesce("pf", "prob")).select("i", "j", "prob")
        print(f"{pf.height} French pairs re-scored with {len(a.frce)} French cross-encoder(s)", flush=True)
    tp.write_parquet(a.save_test_prob)
    s1p, cp = str(a.work / "test_s1" / "*.parquet"), str(a.work / "test_cand" / "*.parquet")
    q = decide(tp.select("i", "j"), tp["prob"].to_numpy(), N[1], N[2])
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "matching_results.tsv", "matched_entity_ids", s1p, cp, q)
    shutil.copy(a.src / "candidate_pairs.tsv", a.out / "candidate_pairs.tsv")
    (a.out / "ce2_band.json").write_text(json.dumps({"arm": nname, "cv": N[0], "threshold": N[1], "exclusive": N[2],
                                                     "control": cname, "control_cv": C[0], "base_cv": v81[0], "frce": bool(a.frce)}, indent=2))
    print(f"wrote {a.out}: {q.height} matches (t={N[1]}, exclusive={N[2]}), {tb.height} test band pairs re-scored; "
          f"test probs -> {a.save_test_prob}", flush=True)


SUBCOMMANDS = {"export-band": main_export_band, "stack": main_stack}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        sys.exit("usage: python rerank.py {" + " | ".join(SUBCOMMANDS) + "} [options]")
    SUBCOMMANDS[sys.argv.pop(1)]()
