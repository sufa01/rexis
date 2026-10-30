"""Validate non-transformer models and export top-N candidates for the ranker (role 3).

For every model:
  1. fit on the train part of the team leave-last-out split, report Recall@10 on the held-out events,
     save candidates to candidates/val_<model>.csv.gz;
  2. refit on all events, save candidates for all users to candidates/test_<model>.csv.gz.
Format: user_id, item_id, score, rank (rank 1 = best), seen items excluded.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.validation import leave_last_out
from src import classic as C

SEED = 42


def build_models() -> dict:
    """Tuned configurations (see EXPERIMENTS.md, section on role 3)."""
    return {
        "popular": lambda: C.Popular(),
        "itemknn": lambda: C.ItemKNN(k=50, shrink=0, alpha=0.7),
        "ease": lambda: C.EASE(reg=400),
        "als": lambda: C.ImplicitMF("als", factors=128, reg=1.0, iterations=20, alpha=5),
        "bpr": lambda: C.ImplicitMF("bpr", factors=64, reg=0.01, iterations=100, lr=0.01),
        "slim": lambda: C.SLIM(l1=1e-3, l2=1e-2, k=100),
        "seq_ease": lambda: C.SeqEASE(reg=400, h_ease=5, horizon=10, decay=0.8, last_n=10, h_last=3, beta=1),
    }


def main():
    models = build_models()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("candidates"))
    parser.add_argument("--models", nargs="+", default=list(models), choices=list(models))
    parser.add_argument("--n-candidates", type=int, default=100)
    parser.add_argument("--no-test", action="store_true", help="only validate, skip the full refit")
    parser.add_argument("--submission", type=str, default=None,
                        help="write submission.csv from this model's full refit")
    args = parser.parse_args()
    np.random.seed(SEED)

    events = pd.read_csv(args.data_dir / "events.csv")
    items = pd.read_csv(args.data_dir / "item_features.csv")["item_id"].to_numpy()
    sample = pd.read_csv(args.data_dir / "submission_sample.csv")
    users = np.sort(events["user_id"].unique())
    train, truth = leave_last_out(events)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds_val = C.Dataset(train, users, items)
    ds_full = C.Dataset(events, users, items)

    results = {}
    for name in args.models:
        t0 = time.time()
        model = models[name]().fit(ds_val)
        cands = C.top_k(model, ds_val, users, k=args.n_candidates)
        r10 = C.recall(cands, truth, 10)
        r100 = C.recall(cands, truth, args.n_candidates)
        cands.to_csv(args.out_dir / f"val_{name}.csv.gz", index=False)
        results[name] = {"recall@10": round(r10, 6), f"recall@{args.n_candidates}": round(r100, 6)}
        print(f"{name:10s} recall@10={r10:.4f} recall@{args.n_candidates}={r100:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        if not args.no_test:
            model = models[name]().fit(ds_full)
            test = C.top_k(model, ds_full, users, k=args.n_candidates)
            test.to_csv(args.out_dir / f"test_{name}.csv.gz", index=False)
            if name == args.submission:
                write_submission(test, sample, Path("submission.csv"))
    truth.to_csv(args.out_dir / "val_truth.csv.gz", index=False)
    (args.out_dir / "val_metrics.json").write_text(json.dumps(results, indent=2))


def write_submission(cands: pd.DataFrame, sample: pd.DataFrame, path: Path, k: int = 10):
    top = cands[cands["rank"] <= k].sort_values(["user_id", "rank"])
    recs = top.groupby("user_id")["item_id"].apply(lambda x: " ".join(map(str, x)))
    out = pd.DataFrame({"user_id": sample["user_id"], "item_id": sample["user_id"].map(recs)})
    assert out["item_id"].str.split().str.len().eq(k).all()
    out.to_csv(path, index=False)
    print(f"Saved submission: {path}")


if __name__ == "__main__":
    main()
