"""Candidate sources that run on the GPU.

Subcommands (python pipeline/candidates.py <subcommand> --help):

  train-retriever
    Fine-tunes multilingual-e5-small as a dense retriever (bi-encoder) with a contrastive InfoNCE loss: for each Source 1
    query one true copy and one hard negative from its blocking candidates, temperature 0.05, 4 epochs, lr 3e-5.
    Trained only on the 99,948 Source 1 records used for the small cross-encoder, which the classifier never trains on,
    so its candidates carry no label leakage.
        python pipeline/candidates.py train-retriever --work work --ce work/ce/ce_train.parquet --out work/retriever

  search
    Top-k Source 2/3 records per Source 1 record within the same country, by cosine similarity of "name | address"
    embeddings (off-the-shelf multilingual-e5-small or the fine-tuned retriever). Exact search on the GPU.
        python pipeline/candidates.py search --work work --split test --model work/retriever --out work/ret
    -> <out>/{split}_knn.parquet: i, j, knn_rank (0-based), knn_cos
"""

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from run import truth_pairs  # noqa: E402


MODEL, MAX_LEN, EPOCHS, LR, TAU = "intfloat/multilingual-e5-small", 64, 4, 3e-5, 0.05


# ── train-retriever ───────────────────────────────────────────────────────────
def main_train_retriever():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--ce", type=Path, default=Path("work/ce/ce_train.parquet"), help="ce_data.py output (i, j, y)")
    ap.add_argument("--batch", type=int, default=256, help="distinct S1 per step (256 fits a 16 GB GPU)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    B = a.batch
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ce = pl.read_parquet(a.ce, columns=["i", "j", "y"])
    s1p, cp = str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")
    pos = truth_pairs(types.SimpleNamespace(data=a.data), s1p, cp).join(ce.select("i").unique(), on="i").select("i", "j")
    neg = ce.filter(pl.col("y") == 0).select("i", "j")
    txt = lambda p, ids: dict(pl.scan_parquet(p).filter(pl.col("idx").is_in(ids.unique().implode())).select(
        "idx", s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).collect().iter_rows())
    ts, tc = txt(s1p, pos["i"]), txt(cp, pl.concat([pos["j"], neg["j"]]))
    P = {k: v for k, v in pos.group_by("i").agg("j").iter_rows()}
    N = {k: v for k, v in neg.group_by("i").agg("j").iter_rows()}
    Ps = {k: set(v) for k, v in P.items()}
    qs = np.array(sorted(P))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda()
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(qs) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler()

    def enc(texts):
        e = tok(["query: " + t for t in texts], max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt").to("cuda")
        h = net(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

    print(f"{len(qs)} S1 queries, {pos.height} positives, {neg.height} hard negatives, {steps} steps", flush=True)
    t0, n = time.time(), 0
    model.train()
    for ep in range(EPOCHS):
        order = rng.permutation(qs)
        for s in range(0, len(order) - B + 1, B):
            q = order[s:s + B]
            p = [int(rng.choice(P[i])) for i in q]
            hn = [int(rng.choice(N[i])) if i in N else int(rng.choice(P[q[(k + 1) % B]])) for k, i in enumerate(q)]
            with torch.autocast("cuda", dtype=torch.float16):
                zq, zc = enc([ts[i] for i in q]), enc([tc[j] for j in p + hn])
                logits = (zq @ zc.T).float() / TAU
            # a batch column that is a true match of row r (other than r's own positive) is not a negative for r:
            # the fallback negative is another query's positive, and a hard negative can be another S1's match
            cand = p + hn
            bad = torch.tensor([[c != r and cand[c] in Ps[i] for c in range(len(cand))] for r, i in enumerate(q)],
                               device="cuda")
            logits = logits.masked_fill(bad, float("-inf"))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(q), device="cuda"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += 1
            if n % 100 == 0:
                print(f"epoch {ep} step {n}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", flush=True)


MAX_LEN = 64


def embed(texts, tok, model, batch=2048):
    """L2-normalised mean-pooled e5 vectors (fp16, kept on CPU), length-sorted batches."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = torch.empty(len(texts), model.module.config.hidden_size if hasattr(model, "module") else model.config.hidden_size,
                      dtype=torch.float16)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for n, s in enumerate(range(0, len(order), batch)):
            k = order[s:s + batch]
            enc = tok(["query: " + texts[x] for x in k], max_length=MAX_LEN, truncation=True, padding=True,
                      return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out[torch.from_numpy(k)] = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).half().cpu()
            if n % 500 == 0:
                print(f"embedded {s}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    return out


# ── search ──────────────────────────────────────────────────────────────────
def main_search():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"), help="normalised records: <work>/{split}_s1, _cand parquet")
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--model", required=True, help="e5 model name or a fine-tuned model folder")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = torch.nn.DataParallel(AutoModel.from_pretrained(a.model).cuda().eval())
    txt = lambda w: (pl.scan_parquet(str(a.work / f"{a.split}_{w}" / "*.parquet"))
                     .select("idx", pl.col("ctry").cast(pl.String),
                             s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
                     .collect())
    s1, cand = txt("s1"), txt("cand")
    res = []
    for ctry in sorted(set(s1["ctry"].unique()) & set(cand["ctry"].unique())):  # open set of countries
        q, c = s1.filter(pl.col("ctry") == ctry), cand.filter(pl.col("ctry") == ctry)
        print(f"{ctry}: {q.height} S1, {c.height} candidates", flush=True)
        C = embed(c["s"].to_list(), tok, model).cuda()
        Q = embed(q["s"].to_list(), tok, model)
        cidx = torch.from_numpy(c["idx"].to_numpy().astype(np.int64)).cuda()
        k = min(a.top, c.height)
        J, S = np.empty((q.height, k), np.uint32), np.empty((q.height, k), np.float16)
        for s in range(0, q.height, 256):
            v, ix = (Q[s:s + 256].cuda() @ C.T).topk(k, dim=1)
            J[s:s + 256], S[s:s + 256] = cidx[ix].cpu().numpy(), v.cpu().numpy()
        del C
        torch.cuda.empty_cache()
        res.append(pl.DataFrame({"i": np.repeat(q["idx"].to_numpy().astype(np.uint32), k), "j": J.ravel(),
                                 "knn_rank": np.tile(np.arange(k, dtype=np.uint8), q.height),
                                 "knn_cos": S.ravel().astype(np.float32)}))
    a.out.mkdir(parents=True, exist_ok=True)
    pl.concat(res).write_parquet(a.out / f"{a.split}_knn.parquet")
    print(f"wrote {a.out / f'{a.split}_knn.parquet'}", flush=True)


SUBCOMMANDS = {"train-retriever": main_train_retriever, "search": main_search}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        sys.exit("usage: python candidates.py {" + " | ".join(SUBCOMMANDS) + "} [options]")
    SUBCOMMANDS[sys.argv.pop(1)]()
