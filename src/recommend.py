from __future__ import annotations
from collections import Counter
import pandas as pd


def popularity_ranking(train: pd.DataFrame, user_col: str = "user_id",
                       item_col: str = "item_id", rating_col: str | None = None,
                       positive_mode: str = "all", threshold: float = 4.0,
                       time_col: str | None = None, half_life_days: float | None = None):
    """
    Global item ranking. Optional rating filter and time decay.
    For time decay, item score is sum(2 ** (-age_days / half_life_days)).
    """
    df = train.copy()
    if positive_mode == "threshold":
        if not rating_col or rating_col not in df.columns:
            raise ValueError("rating_col is required for positive_mode='threshold'")
        df = df[pd.to_numeric(df[rating_col], errors="coerce") >= threshold]
    elif positive_mode != "all":
        raise ValueError("positive_mode must be 'all' or 'threshold'")

    if df.empty:
        return []

    if half_life_days is not None:
        if not time_col or time_col not in df.columns:
            raise ValueError("time_col is required when half_life_days is set")
        times = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        max_time = times.max()
        if pd.isna(max_time):
            raise ValueError("Could not parse any timestamps in time column")
        age_days = (max_time - times).dt.total_seconds().div(86400).clip(lower=0)
        df["_score"] = 2 ** (-age_days / float(half_life_days))
        scores = df.groupby(item_col)["_score"].sum()
    else:
        scores = df.groupby(item_col).size()
    # Deterministic tie-breaking by string representation of item id.
    return sorted(scores.index.tolist(), key=lambda x: (-float(scores.loc[x]), str(x)))


def recommend_for_users(train: pd.DataFrame, users, k: int = 10,
                        user_col: str = "user_id", item_col: str = "item_id",
                        rating_col: str | None = None, positive_mode: str = "all",
                        threshold: float = 4.0, time_col: str | None = None,
                        half_life_days: float | None = None,
                        exclude_seen: bool = True) -> pd.DataFrame:
    ranking = popularity_ranking(
        train, user_col, item_col, rating_col, positive_mode, threshold,
        time_col, half_life_days
    )
    seen = train.groupby(user_col)[item_col].apply(set).to_dict() if exclude_seen else {}
    rows = []
    for user in users:
        blocked = seen.get(user, set())
        selected = [item for item in ranking if item not in blocked][:k]
        # If not enough unseen items exist, do not fabricate item IDs.
        rows.extend((user, item, rank) for rank, item in enumerate(selected, start=1))
    return pd.DataFrame(rows, columns=[user_col, item_col, "rank"])
