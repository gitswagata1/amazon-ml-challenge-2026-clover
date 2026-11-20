"""Cross-encoder pair scorer (GPU): fine-tunes intfloat/multilingual-e5-small (MIT, 118M params; `ce`) or
multilingual-e5-base (MIT, 278M; the re-ranker, two out-of-fold models from ce_data.py --half) to read both
records of a candidate pair together ("name | address" [SEP] "name | address") and predict match / no match.

Trained ONLY on training-set pairs of S1 records outside the matcher's CV sample (ce_train.parquet: i, j, y),
then scores every pair in {split}_pairs.parquet; the score becomes a LightGBM feature (`ce`).
Raw (untransliterated) text is used, so native Indic scripts are read by the multilingual model directly.

python ce.py --text <dir with {split}_s1/cand.parquet> --pairs <dir> --out <dir>
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "intfloat/multilingual-e5-small"
MAX_LEN = 96


def texts(text_dir: Path, split: str, side: str, ids: pl.Series) -> dict:
    """idx -> 'name | address' for the given record ids."""
    t = (pl.scan_parquet(text_dir / f"{split}_{side}.parquet").filter(pl.col("idx").is_in(ids.unique().implode()))
         .select("idx", s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
         .collect())
    return dict(zip(t["idx"].to_list(), t["s"].to_list()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--model", default=MODEL, help="base model (MIT): multilingual-e5-small, or -base for the re-ranker")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=256, help="training pairs per GPU (64 fits e5-base on a 16 GB T4)")
    ap.add_argument("--train-file", default="ce_train.parquet", help="training pairs (i, j, y) inside --pairs")
    ap.add_argument("--model-dir", type=Path, help="score only, with a cross-encoder already trained by this script")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.model_dir:
        tok = AutoTokenizer.from_pretrained(a.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(a.model_dir).cuda().eval()
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).cuda()
        train(a, tok, model)

    score(a, tok, model)


def train(a, tok, model):
    """Fine-tune on <pairs>/<train-file> (training-set pairs only), save to <out>/ce_model.
    Speed: batches are built from length-sorted blocks (little padding) in shuffled order, the next batch is
    tokenised on a CPU thread while the GPU trains, and all visible GPUs are used."""
    from concurrent.futures import ThreadPoolExecutor
    tr = pl.read_parquet(a.pairs / a.train_file)
    ta, tb = texts(a.text, "train", "s1", tr["i"]), texts(a.text, "train", "cand", tr["j"])
    i, j = tr["i"].to_numpy(), tr["j"].to_numpy()
    y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    bs = a.batch * max(1, torch.cuda.device_count())
    lens = np.array([len(ta[x]) + len(tb[z]) for x, z in zip(i, j)])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (tr.height // bs + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    tokenise = lambda k: (k, tok([ta[x] for x in i[k]], [tb[x] for x in j[k]], max_length=MAX_LEN,
                                 truncation=True, padding=True, return_tensors="pt"))
    model.train()
    t0, n = time.time(), 0
    print(f"training {a.model} on {tr.height} pairs, {torch.cuda.device_count()} GPU(s), batch {bs}", flush=True)
    for ep in range(a.epochs):
        rng = np.random.default_rng(ep)
        perm = rng.permutation(tr.height)  # random blocks of 50 batches, length-sorted inside each block
        blocks = [perm[s:s + 50 * bs] for s in range(0, len(perm), 50 * bs)]
        order = np.concatenate([b[np.argsort(lens[b], kind="stable")] for b in blocks])
        chunks = [order[s:s + bs] for s in range(0, len(order), bs)]
        rng.shuffle(chunks)
        with ThreadPoolExecutor(1) as pool:
            nxt = pool.submit(tokenise, chunks[0])
            for c in range(len(chunks)):
                k, enc = nxt.result()
                if c + 1 < len(chunks):
                    nxt = pool.submit(tokenise, chunks[c + 1])
                enc = {m: v.cuda(non_blocking=True) for m, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    logit = net(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y[k].cuda())
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                n += 1
                if n % 500 == 0:
                    print(f"step {n}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    model.save_pretrained(a.out / "ce_model")
    tok.save_pretrained(a.out / "ce_model")


def score(a, tok, model):
    """Score every pair in <pairs>/{split}_pairs.parquet -> <out>/{split}_ce.parquet.
    Throughput: pairs are batched in length order (little padding), the next batch is tokenised on a CPU
    thread while the GPU runs, and all visible GPUs are used. Scores are written back in the input order."""
    from concurrent.futures import ThreadPoolExecutor
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(f"scoring on {torch.cuda.device_count()} GPU(s)", flush=True)
    t0 = time.time()
    for pf in sorted(a.pairs.glob("*_pairs.parquet")):
        split = pf.name.split("_")[0]
        p = pl.read_parquet(pf, columns=["i", "j"])
        ta, tb = texts(a.text, split, "s1", p["i"]), texts(a.text, split, "cand", p["j"])
        i, j = p["i"].to_numpy(), p["j"].to_numpy()
        order = np.argsort([len(ta[x]) + len(tb[y]) for x, y in zip(i, j)], kind="stable")
        bs = 1024 * max(1, torch.cuda.device_count())
        chunks = [order[s:s + bs] for s in range(0, len(order), bs)]
        tokenise = lambda k: (k, tok([ta[x] for x in i[k]], [tb[x] for x in j[k]], max_length=MAX_LEN,
                                     truncation=True, padding=True, return_tensors="pt"))
        out = np.empty(p.height, np.float32)
        with ThreadPoolExecutor(1) as pool, torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            nxt = pool.submit(tokenise, chunks[0]) if chunks else None
            for n in range(len(chunks)):
                k, enc = nxt.result()
                if n + 1 < len(chunks):
                    nxt = pool.submit(tokenise, chunks[n + 1])
                out[k] = model(**{m: v.cuda(non_blocking=True) for m, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy()
                if n % 2000 == 0:
                    print(f"{split}: batch {n}/{len(chunks)} ({time.time() - t0:.0f}s)", flush=True)
        p.with_columns(ce=pl.Series(out)).write_parquet(a.out / f"{split}_ce.parquet")
        print(f"{split}: scored {p.height} pairs ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
