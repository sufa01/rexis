from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.validation import leave_last_out
from src.recommend import recommend_for_users
from src.metrics import recall_at_k_frame


def main():
    parser = argparse.ArgumentParser(description="Evaluate global-popularity baseline with leave-last-out")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--positive-mode", choices=["all", "threshold"], default="all")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--half-life", type=float, default=None,
                        help="Optional decay half-life in timestamp units")
    parser.add_argument("--output", type=Path, default=Path("reports/validation_predictions.csv"))
    args = parser.parse_args()

    events = pd.read_csv(args.data_dir / "events.csv")
    items = pd.read_csv(args.data_dir / "item_features.csv")
    for col in ("user_id", "item_id"):
        if col not in events.columns:
            raise ValueError(f"events.csv must contain {col!r}; found {events.columns.tolist()}")

    time_col = "timestamp" if "timestamp" in events.columns else None
    rating_col = "rating" if "rating" in events.columns else None

    # The target is the latest positive event if threshold mode is used.
    # The full event history is retained for training except the held-out pair.
    if args.positive_mode == "threshold":
        if rating_col is None:
            raise ValueError("threshold mode requires a rating column")
        positive = events[pd.to_numeric(events[rating_col], errors="coerce") >= args.threshold]
        _, truth = leave_last_out(positive, "user_id", "item_id", time_col)
        heldout = set(zip(truth["user_id"], truth["item_id"]))
        train = events[[ (u, i) not in heldout for u, i in zip(events["user_id"], events["item_id"]) ]].copy()
    else:
        train, truth = leave_last_out(events, "user_id", "item_id", time_col)

    users = truth["user_id"].drop_duplicates().tolist()
    recs = recommend_for_users(
        train, users, k=args.k, user_col="user_id", item_col="item_id",
        rating_col=rating_col, positive_mode=args.positive_mode,
        threshold=args.threshold, time_col=time_col, half_life=args.half_life,
        catalog_items=items["item_id"].drop_duplicates().tolist(), exclude_seen=True
    )
    score = recall_at_k_frame(recs, truth, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(args.output, index=False)
    print(f"Recall@{args.k}: {score:.6f}")
    print(f"Validation users: {len(users):,}")
    print(f"Predictions saved: {args.output}")


if __name__ == "__main__":
    main()
