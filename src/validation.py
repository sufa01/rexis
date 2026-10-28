from __future__ import annotations
import pandas as pd


def leave_last_out(events: pd.DataFrame, user_col: str, item_col: str,
                   time_col: str | None = None, seed: int = 42):
    """
    Per-user leave-last-out split. For each user with >=2 distinct interactions,
    hold out the latest row as test and retain earlier rows for training.
    If no time column exists, a seeded shuffle is used only to break ties/order;
    this is not a true temporal validation.
    """
    df = events.copy()
    df["_original_order"] = range(len(df))
    if time_col and time_col in df.columns:
        df = df.sort_values([user_col, time_col, "_original_order"], kind="mergesort")
    else:
        # Preserve source order deterministically; do not pretend it is chronological.
        df = df.sort_values([user_col, "_original_order"], kind="mergesort")

    # Duplicate user-item rows can make the held-out item appear in history.
    # Collapse duplicate pairs by keeping their latest row before splitting.
    df = df.drop_duplicates([user_col, item_col], keep="last")
    counts = df.groupby(user_col)[item_col].transform("size")
    eligible = df[counts >= 2]
    test_idx = eligible.groupby(user_col, sort=False).tail(1).index
    test = df.loc[test_idx, [user_col, item_col]].copy()
    train = df.drop(index=test_idx).copy()
    train = train.drop(columns=["_original_order"], errors="ignore")
    test = test.drop(columns=["_original_order"], errors="ignore")
    return train, test


def positive_events(events: pd.DataFrame, rating_col: str | None = None,
                    positive_mode: str = "all", threshold: float = 4.0) -> pd.DataFrame:
    """positive_mode: 'all' or 'threshold'. No rating column => all events."""
    if positive_mode == "all":
        return events.copy()
    if positive_mode != "threshold":
        raise ValueError("positive_mode must be 'all' or 'threshold'")
    if not rating_col or rating_col not in events.columns:
        raise ValueError(f"rating column {rating_col!r} is required for threshold mode")
    return events.loc[pd.to_numeric(events[rating_col], errors="coerce") >= threshold].copy()
