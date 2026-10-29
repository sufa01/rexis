from __future__ import annotations
import numpy as np
import pandas as pd


def random_holdout(events: pd.DataFrame, frac: float = 1 / 9, seed: int = 42,
                   user_col: str = "user_id"):
    """
    Hold out a random share of every user's events, mimicking the test split.

    In events.csv `timestamp` is a per-user event index with ~10% of positions
    missing, spread uniformly over the history: the hidden test is a random
    ~10% of each user's events, not the latest ones. Holding out n/9 of the
    observed n events reproduces the same share (0.1 of n/0.9).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(events))
    df = events.iloc[order]
    rank = df.groupby(user_col).cumcount()
    n = df.groupby(user_col)[user_col].transform("size")
    n_test = np.maximum(1, np.round(n * frac)).astype(int)
    is_test = (rank < n_test).to_numpy()
    train = df[~is_test].sort_index()
    test = df[is_test].sort_index()
    return train, test


def tail_holdout(events: pd.DataFrame, frac: float = 1 / 9, user_col: str = "user_id",
                 time_col: str = "timestamp"):
    """
    Hold out the latest share of every user's events (n/9 of n, at least one).

    The leaderboard agrees with this split, not with random_holdout: models
    validated at ~0.19 on the random split score 0.094 on the LB and ~0.08-0.10
    here (see EXPERIMENTS.md).
    """
    df = events.sort_values([user_col, time_col], kind="mergesort")
    rank_from_end = df.groupby(user_col).cumcount(ascending=False)
    n = df.groupby(user_col)[user_col].transform("size")
    is_test = (rank_from_end < np.maximum(1, np.round(n * frac))).to_numpy()
    return df[~is_test].sort_index(), df[is_test].sort_index()
