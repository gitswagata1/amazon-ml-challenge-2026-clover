
if __name__ == "__main__":
    # Cross-encoder pseudo-labelling simulation (appended to src/ce.py by ce_pseudo_make.sh).
    # A: e5-small trained on US pairs only. Then continued training (lr 2e-5) from A on
    #   pl:   India pseudo-labels (pos = accepted & p0 >= HI, neg = sample of p0 <= LO) + US replay
    #   ctrl: the same number of US-only pairs (separates "India pseudo-labels" from "more training")
    # Every model scores India's uncertain band (p0 in [0.02, 0.98]); India's real labels are not in the input.
    import glob, os, shutil
    from types import SimpleNamespace
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    HI, LO, RUNS = __HI__, __LO__, __RUNS__
    text = os.path.dirname(sorted(glob.glob("/kaggle/input/**/train_s1.parquet", recursive=True))[0])
    d = os.path.dirname(sorted(glob.glob("/kaggle/input/**/us_train.parquet", recursive=True))[0])
    W = Path("/kaggle/working")
    us = pl.read_parquet(f"{d}/us_train.parquet").select("i", "j", pl.col("y").cast(pl.Int8))
    ind = pl.read_parquet(f"{d}/india.parquet")
    band = ind.filter(pl.col("p0").is_between(0.02, 0.98)).select("i", "j")
    pos = ind.filter(pl.col("acc") & (pl.col("p0") >= HI)).select("i", "j", y=pl.lit(1, pl.Int8))
    neg = ind.filter(pl.col("p0") <= LO).select("i", "j", y=pl.lit(0, pl.Int8))
    neg = neg.sample(min(1_000_000, neg.height), seed=0)
    pl_set = pl.concat([pos, neg, us.sample(500_000, seed=1)]).sample(fraction=1.0, shuffle=True, seed=3)
    print(f"pseudo set: {pos.height} pos + {neg.height} neg + 500000 US replay; band {band.height}", flush=True)

    def run(tag, tr, model_in, lr):
        p = W / f"p_{tag}"
        p.mkdir(exist_ok=True)
        tr.write_parquet(p / "ce_train.parquet")
        band.write_parquet(p / "train_pairs.parquet")
        a = SimpleNamespace(text=Path(text), pairs=p, out=W / tag, epochs=1, model=model_in, lr=lr, batch=256,
                            train_file="ce_train.parquet")
        a.out.mkdir(exist_ok=True)
        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(model_in)
        model = AutoModelForSequenceClassification.from_pretrained(model_in, num_labels=1).cuda()
        train(a, tok, model)
        score(a, tok, model)
        shutil.rmtree(p)
        print(f"{tag} done", flush=True)

    run("A", us, MODEL, 5e-5)
    for r in RUNS:
        tr = pl_set if r == "pl" else us.sample(pl_set.height, with_replacement=pl_set.height > us.height, seed=2)
        run(r, tr, str(W / "A" / "ce_model"), 2e-5)
