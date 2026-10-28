from __future__ import annotations
import pandas as pd


def leave_last_out(events: pd.DataFrame, user_col: str = "user_id",
                   item_col: str = "item_id", time_col: str | None = "timestamp"):
    """
    Hold out the latest unique user-item interaction for each user with >=2 items.
    Sorts by timestamp when available; otherwise preserves file order.
    Duplicate user-item interactions are collapsed to their latest occurrence.
    """
    df = events.copy()
    df["_row_order"] = range(len(df))
    sort_cols = [user_col]
    if time_col and time_col in df.columns:
        sort_cols.append(time_col)
    sort_cols.append("_row_order")
    df = df.sort_values(sort_cols, kind="mergesort")
    df = df.drop_duplicates([user_col, item_col], keep="last")
    counts = df.groupby(user_col)[item_col].transform("nunique")
    eligible = df[counts >= 2]
    test_idx = eligible.groupby(user_col, sort=False).tail(1).index
    test = df.loc[test_idx, [user_col, item_col]].copy()
    train = df.drop(index=test_idx).copy()
    return train.drop(columns=["_row_order"]), test


def positive_events(events: pd.DataFrame, rating_col: str | None = "rating",
                    positive_mode: str = "all", threshold: float = 4.0) -> pd.DataFrame:
    """Filter to all interactions or ratings >= threshold."""
    if positive_mode == "all":
        return events.copy()
    if positive_mode != "threshold":
        raise ValueError("positive_mode must be 'all' or 'threshold'")
    if not rating_col or rating_col not in events.columns:
        raise ValueError(f"Rating column {rating_col!r} is required")
    return events.loc[pd.to_numeric(events[rating_col], errors="coerce") >= threshold].copy()
