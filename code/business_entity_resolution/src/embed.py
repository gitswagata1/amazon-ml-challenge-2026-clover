"""Embedding similarity features for candidate pairs (GPU).

Model: intfloat/multilingual-e5-small (MIT licence, 118M params). Raw names/addresses are embedded as-is,
so native Indic scripts are read by the model directly: a signal independent of our transliteration.
For each (i, j) pair it writes cosine(name_i, name_j) and cosine(addr_i, addr_j); empty text -> NaN.

python embed.py --text <dir with {split}_s1/cand.parquet> --pairs <dir with {split}_pairs.parquet> --out <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "intfloat/multilingual-e5-small"


@torch.no_grad()
def embed(texts, tok, model, max_len, batch=1024):
    out = np.zeros((len(texts), model.config.hidden_size), np.float16)
    order = np.argsort([len(t) for t in texts])  # length-sorted batches = less padding
    for s in range(0, len(texts), batch):
        idx = order[s:s + batch]
        enc = tok(["query: " + texts[k] for k in idx], max_length=max_len, truncation=True, padding=True,
                  return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
        out[idx] = v.float().cpu().numpy().astype(np.float16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    for pf in sorted(a.pairs.glob("*_pairs.parquet")):
        split = pf.name.split("_")[0]
        p = pl.read_parquet(pf, columns=["i", "j"])
        res = {"i": p["i"], "j": p["j"]}
        for field, col, max_len in (("name", "business_name", 32), ("addr", "business_address", 64)):
            cos = []
            for side, key in (("s1", "i"), ("cand", "j")):
                ids = p[key].unique().sort()
                t = (pl.scan_parquet(a.text / f"{split}_{side}.parquet").select("idx", col)
                     .filter(pl.col("idx").is_in(ids.implode())).collect().sort("idx"))
                e = embed(t[col].fill_null("").to_list(), tok, model, max_len)
                e[(t[col].fill_null("") == "").to_numpy()] = np.nan
                cos.append((t["idx"].to_numpy(), e))
                print(f"{split} {field} {side}: {len(t)} texts embedded", flush=True)
            (ia, ea), (ja, eb) = cos
            ri = np.searchsorted(ia, p["i"].to_numpy())
            rj = np.searchsorted(ja, p["j"].to_numpy())
            sims = np.empty(p.height, np.float32)
            for s in range(0, p.height, 5_000_000):  # chunked row-wise dot products
                sims[s:s + 5_000_000] = (ea[ri[s:s + 5_000_000]].astype(np.float32)
                                         * eb[rj[s:s + 5_000_000]].astype(np.float32)).sum(1)
            res[f"emb_{field}"] = sims
            del cos, ea, eb
        pl.DataFrame(res).write_parquet(a.out / f"{split}_emb.parquet")
        print(f"{split}: wrote {p.height} pair similarities", flush=True)


if __name__ == "__main__":
    main()
