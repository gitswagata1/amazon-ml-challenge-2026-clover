"""Idea 1b: name-rarity features (frequency-based Fellegi-Sunter) on the A/B table, tuned LightGBM, 5-fold by S1.
name_df_s1 / name_df_cand = number of S1 / candidates in the same country with the same exact core name,
cand_addr_empty, and name_df_s1 x cand_addr_empty. Control = the same table without them (0.95287).

python experiments/rarity_ab.py
"""
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from ab import TUNED
from tune import macro_f05

D = "work/ab_data"
s1 = (pl.scan_parquet("work/train_s1/*.parquet").select(i="idx", ctry="ctry", core="core").collect()
      .with_columns(name_df_s1=pl.len().over("ctry", "core").cast(pl.Float32)))
c = (pl.scan_parquet("work/train_cand/*.parquet")
     .select(j="idx", ctry="ctry", core="core", cand_addr_empty=(pl.col("business_address").fill_null("").str.strip_chars() == "").cast(pl.Int8))
     .collect().with_columns(name_df_cand=pl.len().over("ctry", "core").cast(pl.Float32)))
df = (pl.read_parquet(f"{D}/train_features.parquet")
      .drop([x for x in ("r_rel_c", "r_gap_c", "sup_sk", "sup_hn", "sup_both") if x in pl.read_parquet_schema(f"{D}/train_features.parquet")])
      .join(pl.read_parquet(f"{D}/train_tfidf.parquet"), on=["i", "j"], how="left")
      .join(pl.read_parquet(f"{D}/train_emb_v3pairs.parquet"), on=["i", "j"], how="left")
      .join(s1.select("i", "name_df_s1"), on="i", how="left")
      .join(c.select("j", "name_df_cand", "cand_addr_empty"), on="j", how="left")
      .with_columns(df_s1_x_empty=pl.col("name_df_s1") * pl.col("cand_addr_empty")).sort("i"))
meta = pl.read_parquet(f"{D}/tune_meta.parquet")
X, y, g = df.drop("i", "j", "y").to_numpy().astype(np.float32), df["y"].to_numpy(), df["i"].to_numpy()
oof = np.zeros(len(y))
for tr, va in GroupKFold(5).split(X, y, g):
    oof[va] = lgb.LGBMClassifier(**TUNED).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
s, t = macro_f05(df, oof, meta)
print(f"+ rarity features: CV macro F0.5 = {s:.5f} @ t={t}  (control 0.95287)", flush=True)
