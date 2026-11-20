"""Optuna search over LightGBM hyper-parameters on a saved feature table.

Inputs come from `run.py train --save-features`:
  train_features.parquet  features + i, j, y (one row per candidate pair)
  tune_meta.parquet       i, ntrue for every sampled S1 record (true match count, incl. unblocked ones)
Objective = the challenge metric: macro F0.5 (singletons included) after the one-to-one rule and the best threshold.

python tune.py --dir <folder with both files> --trials 50 --out best_params.json
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.model_selection import GroupKFold


def macro_f05(df: pl.DataFrame, prob, meta: pl.DataFrame, thresholds=np.arange(0.3, 0.92, 0.02)):
    """Best macro F0.5 over thresholds -> (score, threshold). df: i, j, y aligned with prob."""
    q = df.select("i", "j", "y").with_columns(prob=pl.Series(prob))
    q = q.filter(pl.col("prob") == pl.col("prob").max().over("j"))
    best = (-1.0, 0.5)
    for t in thresholds:
        per = (meta.join(q.filter(pl.col("prob") >= t).group_by("i").agg(npred=pl.len(), tp=pl.col("y").sum()),
                         on="i", how="left").fill_null(0))
        P, R = pl.col("tp") / pl.col("npred"), pl.col("tp") / pl.col("ntrue")
        s = per.select(pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0)
                       .when(pl.col("tp") == 0).then(0.0)
                       .otherwise(1.25 * P * R / (0.25 * P + R)).mean()).item()
        best = max(best, (s, round(float(t), 2)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("work"))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--n-s1", type=int, default=150_000, help="S1 records used for tuning (subsample)")
    ap.add_argument("--out", type=Path, default=Path("best_params.json"))
    ap.add_argument("--hours", type=float, default=10.0, help="stop starting new trials after this many hours")
    a = ap.parse_args()

    meta = pl.read_parquet(a.dir / "tune_meta.parquet")
    meta = meta.sample(min(a.n_s1, meta.height), seed=1)
    df = pl.read_parquet(a.dir / "train_features.parquet").join(meta.select("i"), on="i").sort("i")
    X = df.drop("i", "j", "y").to_numpy().astype(np.float32)
    y, groups = df["y"].to_numpy(), df["i"].to_numpy()
    folds = list(GroupKFold(a.folds).split(X, y, groups))
    print(f"{meta.height} S1, {len(y)} pairs, {X.shape[1]} features", flush=True)

    def objective(trial):
        params = dict(
            objective="binary", verbose=-1, n_jobs=-1, feature_pre_filter=False,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 500, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0), bagging_freq=1,
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 0.5, 2.0, log=True),
        )
        oof, rounds = np.zeros(len(y)), []
        for k, (tr, va) in enumerate(folds):
            dtr, dva = lgb.Dataset(X[tr], y[tr]), lgb.Dataset(X[va], y[va])
            m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
            rounds.append(m.best_iteration)
            fold = df[va]
            s, _ = macro_f05(fold, oof[va], meta.join(fold.select("i").unique(), on="i"))
            trial.report(s, k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        score, t = macro_f05(df, oof, meta)
        trial.set_user_attr("threshold", t)
        trial.set_user_attr("n_estimators", int(np.mean(rounds)))
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    # baseline = current run.py PARAMS, so tuning can only report an improvement over it
    study.enqueue_trial(dict(learning_rate=0.05, num_leaves=127, min_child_samples=50, feature_fraction=0.8,
                             bagging_fraction=0.8, lambda_l1=1e-3, lambda_l2=1e-3, scale_pos_weight=1.0))
    study.optimize(objective, n_trials=a.trials, timeout=a.hours * 3600, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.value} best={s.best_value:.5f}", flush=True)])
    best = study.best_trial
    out = {**best.params, "n_estimators": best.user_attrs["n_estimators"], "cv_f05": best.value,
           "threshold": best.user_attrs["threshold"], "baseline_f05": study.trials[0].value}
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
