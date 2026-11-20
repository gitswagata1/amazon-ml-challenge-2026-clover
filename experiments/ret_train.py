"""Fine-tuned retriever (Kaggle GPU): contrastive bi-encoder on multilingual-e5-small (MIT), trained ONLY on the
cross-encoder's 99,948 excluded S1 (none of the matcher's training S1), so its kNN candidates never leak labels.

Each step: B distinct S1 queries, for each one random true copy (positive) and one random blocking candidate that is
not a match (hard negative). InfoNCE over the batch's 2B candidates (temperature 0.05); one S1 per batch, so
another row's candidate can never be a true match. Output: /kaggle/working/retriever (for n4_search.py MODEL).
"""
import glob
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

MODEL, MAX_LEN, B, EPOCHS, LR, TAU = "intfloat/multilingual-e5-small", 64, 256, 4, 3e-5, 0.05


def find(name):
    return sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))[0]


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pos, neg = pl.read_parquet(find("pos.parquet")), pl.read_parquet(find("neg.parquet"))
    txt = lambda f, ids: dict(pl.scan_parquet(find(f)).filter(pl.col("idx").is_in(ids.unique().implode())).select(
        "idx", s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).collect().iter_rows())
    ts, tc = txt("train_s1.parquet", pos["i"]), txt("train_cand.parquet", pl.concat([pos["j"], neg["j"]]))
    P = {k: v for k, v in pos.group_by("i").agg("j").iter_rows()}
    N = {k: v for k, v in neg.group_by("i").agg("j").iter_rows()}
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
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(q), device="cuda"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += 1
            if n % 100 == 0:
                print(f"epoch {ep} step {n}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained("/kaggle/working/retriever")
    tok.save_pretrained("/kaggle/working/retriever")
    print("saved", flush=True)


if __name__ == "__main__":
    main()
