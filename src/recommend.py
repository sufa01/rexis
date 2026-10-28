from __future__ import annotations
import pandas as pd


def popularity_ranking(train: pd.DataFrame, item_col: str = "item_id",
                       rating_col: str | None = "rating", positive_mode: str = "all",
                       threshold: float = 4.0, time_col: str | None = None,
                       half_life: float | None = None, catalog_items=None) -> list:
    """
    Return deterministic item ranking by interaction count.
    Optional threshold mode counts only ratings >= threshold.
    Optional time decay uses timestamp units (the supplied dataset's timestamp
    is an event-order index, so half-life is measured in events, not days).
    """
    df = train.copy()
    if positive_mode == "threshold":
        if not rating_col or rating_col not in df.columns:
            raise ValueError("rating_col is required for positive_mode='threshold'")
        df = df[pd.to_numeric(df[rating_col], errors="coerce") >= threshold]
    elif positive_mode != "all":
        raise ValueError("positive_mode must be 'all' or 'threshold'")

    if half_life is not None:
        if not time_col or time_col not in df.columns:
            raise ValueError("time_col is required when half_life is set")
        times = pd.to_numeric(df[time_col], errors="coerce")
        max_time = times.max()
        age = (max_time - times).clip(lower=0)
        df["_score"] = 2.0 ** (-age / float(half_life))
        scores = df.groupby(item_col)["_score"].sum()
    else:
        scores = df.groupby(item_col).size().astype(float)

    known = scores.to_dict()
    candidates = list(catalog_items) if catalog_items is not None else list(scores.index)
    # All catalogue items are candidates; unseen items have score 0.
    candidates = list(dict.fromkeys(candidates))
    return sorted(candidates, key=lambda item: (-float(known.get(item, 0.0)), str(item)))


def recommend_for_users(train: pd.DataFrame, users, k: int = 10,
                        user_col: str = "user_id", item_col: str = "item_id",
                        rating_col: str | None = "rating", positive_mode: str = "all",
                        threshold: float = 4.0, time_col: str | None = None,
                        half_life: float | None = None, catalog_items=None,
                        exclude_seen: bool = True) -> pd.DataFrame:
    ranking = popularity_ranking(
        train, item_col=item_col, rating_col=rating_col, positive_mode=positive_mode,
        threshold=threshold, time_col=time_col, half_life=half_life,
        catalog_items=catalog_items
    )
    seen = train.groupby(user_col)[item_col].apply(set).to_dict() if exclude_seen else {}
    rows = []
    for user in users:
        blocked = seen.get(user, set())
        selected = [item for item in ranking if item not in blocked][:k]
        rows.extend((user, item, rank) for rank, item in enumerate(selected, start=1))
    return pd.DataFrame(rows, columns=[user_col, item_col, "rank"])
