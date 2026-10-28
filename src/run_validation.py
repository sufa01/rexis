from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.validation import leave_last_out
from src.recommend import recommend_for_users
from src.metrics import recall_at_k_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rating-col", default="rating")
    p.add_argument("--positive-mode", choices=["all", "threshold"], default="all")
    p.add_argument("--threshold", type=float, default=4.0)
    p.add_argument("--time-col", default="timestamp")
    p.add_argument("--half-life-days", type=float, default=None)
    p.add_argument("--output", type=Path, default=Path("reports/validation_predictions.csv"))
    args = p.parse_args()

    events = pd.read_csv(args.data_dir / "events.csv")
    for col in ("user_id", "item_id"):
        if col not in events.columns:
            raise ValueError(f"events.csv must contain {col!r}; found {events.columns.tolist()}")
    rating_col = args.rating_col if args.rating_col in events.columns else None
    time_col = args.time_col if args.time_col in events.columns else None

    # For threshold-based evaluation, filter positives before splitting so the held-out
    # target is guaranteed to be a positive event. The history still comes from all events.
    if args.positive_mode == "threshold":
        if not rating_col:
            raise ValueError("threshold mode requested but rating column is missing")
        positives = events[pd.to_numeric(events[rating_col], errors="coerce") >= args.threshold]
        _, truth = leave_last_out(positives, "user_id", "item_id", time_col)
        # Keep each target user's full history except the held-out target item.
        train = events[~events.set_index(["user_id", "item_id"]).index.isin(
            truth.set_index(["user_id", "item_id"]).index
        )].copy()
    else:
        train, truth = leave_last_out(events, "user_id", "item_id", time_col)

    users = truth["user_id"].drop_duplicates().tolist()
    recs = recommend_for_users(
        train, users, k=args.k, user_col="user_id", item_col="item_id",
        rating_col=rating_col, positive_mode=args.positive_mode,
        threshold=args.threshold, time_col=time_col,
        half_life_days=args.half_life_days, exclude_seen=True
    )
    score = recall_at_k_frame(recs, truth, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(args.output, index=False)
    print(f"Recall@{args.k}: {score:.6f}")
    print(f"Validation users: {len(users):,}")
    print(f"Predictions saved: {args.output}")


if __name__ == "__main__":
    main()
