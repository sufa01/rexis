from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.recommend import recommend_for_users


def find_col(df, expected, alternatives=()):
    if expected in df.columns:
        return expected
    for col in alternatives:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {expected!r}. Available columns: {df.columns.tolist()}")


def main():
    p = argparse.ArgumentParser(description="Evaluate popularity baseline and/or create submission CSV")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--sample", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("submission.csv"))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rating-col", default="rating")
    p.add_argument("--positive-mode", choices=["all", "threshold"], default="all")
    p.add_argument("--threshold", type=float, default=4.0)
    p.add_argument("--time-col", default="timestamp")
    p.add_argument("--half-life-days", type=float, default=None)
    args = p.parse_args()

    data_dir = args.data_dir
    events = pd.read_csv(data_dir / "events.csv")
    sample_path = args.sample or data_dir / "submission_sample.csv"
    sample = pd.read_csv(sample_path) if sample_path.exists() else None
    user_col = find_col(events, "user_id", ("user",))
    item_col = find_col(events, "item_id", ("item",))
    rating_col = args.rating_col if args.rating_col in events.columns else None
    time_col = args.time_col if args.time_col in events.columns else None

    if sample is not None:
        sample_user = find_col(sample, "user_id", ("user",))
        sample_item = find_col(sample, "item_id", ("item",))
        # If sample contains one row per recommendation slot, use unique users.
        users = sample[sample_user].drop_duplicates().tolist()
        user_col_out, item_col_out = sample_user, sample_item
    else:
        users = events[user_col].drop_duplicates().tolist()
        user_col_out, item_col_out = user_col, item_col

    recs = recommend_for_users(
        events, users, k=args.k, user_col=user_col, item_col=item_col,
        rating_col=rating_col, positive_mode=args.positive_mode,
        threshold=args.threshold, time_col=time_col,
        half_life_days=args.half_life_days, exclude_seen=True
    )
    recs = recs.rename(columns={user_col: user_col_out, item_col: item_col_out})
    recs = recs[[user_col_out, item_col_out]].copy()

    # Exact required output schema and string IDs.
    recs[user_col_out] = recs[user_col_out].astype(str)
    recs[item_col_out] = recs[item_col_out].astype(str)
    recs.to_csv(args.output, index=False)
    print(f"Saved {len(recs):,} recommendations to {args.output}")
    print(f"Users: {recs[user_col_out].nunique():,}; rows per user (min/max): "
          f"{recs.groupby(user_col_out).size().min()}/{recs.groupby(user_col_out).size().max()}")
    if sample is not None:
        print(f"Output columns: {recs.columns.tolist()}; sample columns: {[user_col_out, item_col_out]}")


if __name__ == "__main__":
    main()
