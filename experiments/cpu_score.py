# CPU scoring with a fine-tuned cross-encoder, identical input format to ce.py ("name | address" pair, max_len 96, logit)
# (used for the 210,562 v10 French pairs outside the Kaggle P5 run: experiments/cpu_score.py <frce ce_model> work/text <pairs> <out>)
# usage: PYTHONPATH=<dir with transformers==4.57.6> python experiments/cpu_score.py MODEL_DIR TEXT_DIR PAIRS OUT [--check REF]
import sys, time
import numpy as np, polars as pl, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_dir, text_dir, pairs, out = sys.argv[1:5]
split = "test"
if "--split" in sys.argv: split = sys.argv[sys.argv.index("--split") + 1]
ref = sys.argv[6] if len(sys.argv) > 6 and sys.argv[5] == "--check" else None
torch.set_num_threads(30)
p = pl.read_parquet(pairs, columns=["i", "j"])
if ref:
    p = pl.read_parquet(ref).sample(2000, seed=0).select("i", "j")
txt = lambda f, ids: dict(pl.scan_parquet(f"{text_dir}/{f}").filter(pl.col("idx").is_in(ids.unique().implode()))
    .select("idx", s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).collect().iter_rows())
ta, tb = txt(f"{split}_s1.parquet", p["i"]), txt(f"{split}_cand.parquet", p["j"])
tok = AutoTokenizer.from_pretrained(model_dir)
m = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
i, j = p["i"].to_list(), p["j"].to_list()
lens = np.array([len(ta[x]) + len(tb[y]) for x, y in zip(i, j)]); order = np.argsort(lens)
out_s = np.zeros(len(i), np.float32); t0 = time.time()
with torch.inference_mode():
    for s in range(0, len(order), 256):
        k = order[s:s + 256]
        enc = tok([ta[i[x]] for x in k], [tb[j[x]] for x in k], max_length=96, truncation=True, padding=True, return_tensors="pt")
        out_s[k] = m(**enc).logits.squeeze(-1).float().numpy()
        if s % 25600 == 0:
            print(f"{s}/{len(i)} ({time.time() - t0:.0f}s)", flush=True)
res = p.with_columns(ce=pl.Series(out_s))
if ref:
    r = pl.read_parquet(ref).join(res, on=["i", "j"], suffix="_cpu")
    print("check vs Kaggle: pearson", round(float(np.corrcoef(r["ce"], r["ce_cpu"])[0, 1]), 6), "max|diff|", round(float((r["ce"] - r["ce_cpu"]).abs().max()), 4))
else:
    res.write_parquet(out); print("wrote", out, res.height, f"({time.time() - t0:.0f}s)")
