"""N4 test (Kaggle GPU): would embedding nearest-neighbour search find the true matches our key blocking missed?

Embeds every training candidate record and every S1 with a missed true match (e5-small, "name | address"),
then for each such S1 takes the top-50 records of its country by cosine and writes the rank of each missed
true match (0-based, -1 = not in top 50) -> /kaggle/working/n4_rank.parquet (i, j, ctry, rank).
"""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

MODEL, MAX_LEN, TOP = "intfloat/multilingual-e5-small", 64, 50


def find(name):
    return sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))[0]


def embed(texts, tok, model, batch=2048):
    """L2-normalised mean-pooled e5 vectors (fp16, CPU), length-sorted batches."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = torch.empty(len(texts), 384, dtype=torch.float16)
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


def main():
    missed = pl.read_parquet(find("missed.parquet"))
    cc = pl.read_parquet(find("cand_ctry.parquet")).with_columns(pl.col("ctry").cast(pl.String))
    txt = lambda f, ids: (pl.scan_parquet(find(f)).filter(pl.col("idx").is_in(ids.implode()))
                          .select("idx", s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
                          .collect())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = torch.nn.DataParallel(AutoModel.from_pretrained(MODEL).cuda().eval())
    res = []
    for ctry in ("us", "india"):
        m = missed.filter(pl.col("ctry") == ctry)
        c = txt("train_cand.parquet", cc.filter(pl.col("ctry") == ctry)["idx"])
        q = txt("train_s1.parquet", m["i"].unique())
        print(f"{ctry}: {c.height} candidate records, {q.height} S1 queries", flush=True)
        C = embed(c["s"].to_list(), tok, model).cuda()
        Q = embed(q["s"].to_list(), tok, model)
        cidx = torch.from_numpy(c["idx"].to_numpy().astype(np.int64)).cuda()
        top = np.empty((q.height, TOP), np.int64)
        for s in range(0, q.height, 256):
            sim = Q[s:s + 256].cuda() @ C.T
            top[s:s + 256] = cidx[sim.topk(TOP, dim=1).indices].cpu().numpy()
        del C
        torch.cuda.empty_cache()
        pos = {i: r for r, i in enumerate(q["idx"].to_list())}
        ranks = []
        for i, j in zip(m["i"].to_list(), m["j"].to_list()):
            hit = np.nonzero(top[pos[i]] == j)[0]
            ranks.append(int(hit[0]) if len(hit) else -1)
        res.append(m.with_columns(rank=pl.Series(ranks, dtype=pl.Int16)))
        r = np.array(ranks)
        print(f"{ctry}: missed matches recovered in top 5/10/20/50: "
              + " / ".join(f"{((r >= 0) & (r < k)).mean():.3f}" for k in (5, 10, 20, 50)), flush=True)
    pl.concat(res).write_parquet("/kaggle/working/n4_rank.parquet")


if __name__ == "__main__":
    main()
