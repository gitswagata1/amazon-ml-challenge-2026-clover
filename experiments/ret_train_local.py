"""Fine-tuned retriever, LOCAL version for a small GPU (e.g. RTX 3050 4 GB): same as the Kaggle amizon-ret-train
kernel (contrastive bi-encoder on multilingual-e5-small, MIT; trained ONLY on the cross-encoder's 99,948 excluded S1,
so its kNN candidates never leak matcher labels), with a smaller batch + gradient checkpointing to fit in 4 GB.

Inputs in --data: pos.parquet, neg.parquet (Kaggle dataset kaggle-B/amizon-ret-data-w),
                  train_s1.parquet, train_cand.parquet (kaggle-B/amizon-text-w).
Output: --out folder with the model (config.json, model.safetensors, tokenizer files).

python ret_train_local.py --data ret_data --out amizon-retriever-local
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

MODEL, MAX_LEN, TAU = "intfloat/multilingual-e5-small", 64, 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128, help="distinct S1 per step (256 on a 16 GB GPU)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pos, neg = pl.read_parquet(a.data / "pos.parquet"), pl.read_parquet(a.data / "neg.parquet")
    txt = lambda f, ids: dict(pl.scan_parquet(a.data / f).filter(pl.col("idx").is_in(ids.unique().implode())).select(
        "idx", s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).collect().iter_rows())
    ts, tc = txt("train_s1.parquet", pos["i"]), txt("train_cand.parquet", pl.concat([pos["j"], neg["j"]]))
    P = {k: v for k, v in pos.group_by("i").agg("j").iter_rows()}
    N = {k: v for k, v in neg.group_by("i").agg("j").iter_rows()}
    qs = np.array(sorted(P))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda()
    model.gradient_checkpointing_enable()          # activations recomputed in backward: fits 4 GB
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    B = a.batch
    steps = a.epochs * (len(qs) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler()

    def enc(texts):
        e = tok(["query: " + t for t in texts], max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt").to("cuda")
        h = model(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

    print(f"{len(qs)} S1 queries, {pos.height} positives, {neg.height} hard negatives, {steps} steps, "
          f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    t0, n = time.time(), 0
    model.train()
    for ep in range(a.epochs):
        order = rng.permutation(qs)
        for s in range(0, len(order) - B + 1, B):
            q = order[s:s + B]
            p = [int(rng.choice(P[i])) for i in q]
            hn = [int(rng.choice(N[i])) if i in N else int(rng.choice(P[q[(k + 1) % B]])) for k, i in enumerate(q)]
            with torch.autocast("cuda", dtype=torch.float16):
                zq, zc = enc([ts[i] for i in q]), enc([tc[j] for j in p + hn])
                logits = (zq @ zc.T).float() / TAU
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(q), device="cuda"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += 1
            if n % 50 == 0:
                print(f"epoch {ep} step {n}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s, "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved to {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
