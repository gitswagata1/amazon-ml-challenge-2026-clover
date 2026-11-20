"""Cache the unseen-country proxies used for France rules (v11 mining): LightGBM without the ce feature, trained on one
train country (250k S1 sample), scoring the other country's S1 (all India / 600k US). Writes work9/proxy_<src>2<dst>.parquet
(i, j, y, prob). The US->India proxy matched the France LB probe (P1: +0.0059 simulated vs +0.0075 on France).

nice -n 15 python experiments/proxy_cache.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # in the submission zip: src/experiments/ -> src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))  # git repo
from run import PARAMS  # noqa: E402

W = Path("work9")


def load(ids):
    df = pl.scan_parquet(W / "train_features.parquet").filter(pl.col("i").is_in(ids.implode())).collect()
    for pat in ("gpu/train_emb.parquet", "tfw_train.parquet", "rarity_train.parquet"):
        e = pl.scan_parquet(W / pat).filter(pl.col("i").is_in(ids.implode())).collect()
        df = df.join(e, on=["i", "j"], how="left", maintain_order="left")
    return df


def main():
    t = json.loads(Path("code/business_entity_resolution/best_params.json").read_text())
    PARAMS.update(num_leaves=t["num_leaves"], min_child_samples=t["min_child_samples"], colsample_bytree=t["feature_fraction"],
                  subsample=t["bagging_fraction"], reg_alpha=t["lambda_l1"], reg_lambda=t["lambda_l2"],
                  scale_pos_weight=t["scale_pos_weight"], n_estimators=500, learning_rate=0.08, n_jobs=8)
    cty = pl.scan_parquet(str(W / "train_s1" / "*.parquet")).select(i="idx", c="country").collect()
    have = pl.scan_parquet(W / "train_features.parquet").select("i").unique().collect().join(cty, on="i")
    for src, dst, n_dst in (("US", "India", None), ("India", "US", 600000)):
        tr = load(have.filter(pl.col("c") == src).sample(250000, seed=0)["i"])
        cols = [c for c in tr.columns if c not in ("i", "j", "y")]
        m = lgb.LGBMClassifier(**PARAMS).fit(tr.select(cols).to_pandas(), tr["y"].to_numpy())
        del tr
        d = have.filter(pl.col("c") == dst)
        d = d if n_dst is None else d.sample(n_dst, seed=1)
        te = load(d["i"])
        out = te.select("i", "j", "y").with_columns(prob=pl.Series(m.predict_proba(te.select(cols).to_pandas())[:, 1], dtype=pl.Float32))
        f = W / f"proxy_{src.lower()}2{dst.lower()}.parquet"
        out.write_parquet(f)
        print(f"wrote {f}: {out['i'].n_unique()} S1, {out.height} pairs", flush=True)
        del te, out


if __name__ == "__main__":
    main()
