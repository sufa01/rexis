from __future__ import annotations
import numpy as np
import pandas as pd


def recall_at_k(recommendations: dict, truth: dict, k: int = 10) -> float:
    """Macro Recall@k over users that have at least one relevant held-out item."""
    scores = []
    for user, relevant in truth.items():
        relevant = set(relevant)
        if relevant:
            scores.append(len(set(recommendations.get(user, [])[:k]) & relevant) / len(relevant))
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k_frame(recommendations: pd.DataFrame, truth: pd.DataFrame, k: int = 10,
                      user_col: str = "user_id", item_col: str = "item_id") -> float:
    recs = recommendations.groupby(user_col)[item_col].apply(list).to_dict()
    targets = truth.groupby(user_col)[item_col].apply(list).to_dict()
    return recall_at_k(recs, targets, k)
