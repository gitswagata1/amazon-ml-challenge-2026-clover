"""N4 candidate search (Kaggle GPU): for every S1 of SPLIT, the TOP nearest S2/S3 records of the same country by
cosine of e5-small "name | address" embeddings (MIT model, raw text only).
-> /kaggle/working/{SPLIT}_knn.parquet: i, j, knn_rank (0-based), knn_cos. Merged into the pairs on the workstation.
"""
import glob
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

SPLIT, MODEL, MAX_LEN, TOP = "__SPLIT__", "intfloat/multilingual-e5-small", 64, 20


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
    global MODEL  # the fine-tuned retriever from the amizon-ret-train kernel output
    MODEL = __import__("os").path.dirname(sorted(f for f in glob.glob("/kaggle/input/**/config.json", recursive=True) if "retriever" in f)[0])
    print("model", MODEL, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = torch.nn.DataParallel(AutoModel.from_pretrained(MODEL).cuda().eval())
    txt = lambda w: (pl.read_parquet(find(f"{SPLIT}_{w}.parquet"))
                     .select("idx", s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
                     .join(pl.read_parquet(find(f"{SPLIT}_{w}_ctry.parquet")).with_columns(pl.col("ctry").cast(pl.String)), on="idx"))
    s1, cand = txt("s1"), txt("cand")
    res = []
    for ctry in sorted(set(s1["ctry"].unique()) & set(cand["ctry"].unique())):  # open set of countries
        q, c = s1.filter(pl.col("ctry") == ctry), cand.filter(pl.col("ctry") == ctry)
        print(f"{ctry}: {q.height} S1, {c.height} candidates", flush=True)
        C = embed(c["s"].to_list(), tok, model).cuda()
        Q = embed(q["s"].to_list(), tok, model)
        cidx = torch.from_numpy(c["idx"].to_numpy().astype(np.int64)).cuda()
        k = min(TOP, c.height)
        J, S = np.empty((q.height, k), np.uint32), np.empty((q.height, k), np.float16)
        for s in range(0, q.height, 256):
            v, ix = (Q[s:s + 256].cuda() @ C.T).topk(k, dim=1)
            J[s:s + 256], S[s:s + 256] = cidx[ix].cpu().numpy(), v.cpu().numpy()
        del C
        torch.cuda.empty_cache()
        res.append(pl.DataFrame({"i": np.repeat(q["idx"].to_numpy().astype(np.uint32), k), "j": J.ravel(),
                                 "knn_rank": np.tile(np.arange(k, dtype=np.uint8), q.height),
                                 "knn_cos": S.ravel().astype(np.float32)}))
        print(f"{ctry}: done", flush=True)
    pl.concat(res).write_parquet(f"/kaggle/working/{SPLIT}_knn.parquet")


if __name__ == "__main__":
    main()
