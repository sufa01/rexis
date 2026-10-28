from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.recommend import recommend_for_users


def main():
    parser = argparse.ArgumentParser(description="Generate submission in the supplied sample format")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--positive-mode", choices=["all", "threshold"], default="all")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--half-life", type=float, default=None,
                        help="Optional time-decay half-life in timestamp units")
    args = parser.parse_args()

    events = pd.read_csv(args.data_dir / "events.csv")
    sample_path = args.sample or args.data_dir / "submission_sample.csv"
    sample = pd.read_csv(sample_path)
    items = pd.read_csv(args.data_dir / "item_features.csv")

    required_events = {"user_id", "item_id"}
    if not required_events.issubset(events.columns):
        raise ValueError(f"events.csv must contain {required_events}; found {events.columns.tolist()}")
    if not {"user_id", "item_id"}.issubset(sample.columns):
        raise ValueError("submission_sample.csv must contain user_id and item_id columns")
    if "item_id" not in items.columns:
        raise ValueError("item_features.csv must contain item_id")

    users = sample["user_id"].drop_duplicates().tolist()
    catalog = items["item_id"].drop_duplicates().tolist()
    rating_col = "rating" if "rating" in events.columns else None
    time_col = "timestamp" if "timestamp" in events.columns else None
    recs = recommend_for_users(
        events, users, k=args.k, user_col="user_id", item_col="item_id",
        rating_col=rating_col, positive_mode=args.positive_mode,
        threshold=args.threshold, time_col=time_col, half_life=args.half_life,
        catalog_items=catalog, exclude_seen=True
    )

    # The provided sample has ONE ROW PER USER; item_id is a space-separated
    # string containing exactly 10 item IDs, not one row per recommendation.
    grouped = recs.sort_values(["user_id", "rank"]).groupby("user_id")["item_id"].apply(
        lambda values: " ".join(str(x) for x in values)
    )
    output = pd.DataFrame({"user_id": users})
    output["item_id"] = output["user_id"].map(grouped).fillna("")
    output["user_id"] = output["user_id"].astype(str)
    output["item_id"] = output["item_id"].astype(str)

    # Match sample schema and row order, including any user with no available recs.
    output = output[["user_id", "item_id"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    lengths = output["item_id"].str.split().str.len()
    print(f"Saved: {args.output}")
    print(f"Users: {len(output)}; rows with exactly {args.k} recommendations: {(lengths == args.k).sum()}")
    print(f"Output columns: {output.columns.tolist()}")
    if (lengths != args.k).any():
        print("WARNING: some users have fewer than k recommendations; check catalogue size and exclusions.")


if __name__ == "__main__":
    main()
