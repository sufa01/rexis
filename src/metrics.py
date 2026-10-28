from __future__ import annotations
import numpy as np
import pandas as pd


def recall_at_k(recommendations: dict, truth: dict, k: int = 10) -> float:
    """Macro Recall@k. Users with no held-out positives are excluded."""
    scores = []
    for user, relevant in truth.items():
        relevant = set(relevant)
        if not relevant:
            continue
        recs = list(recommendations.get(user, []))[:k]
        scores.append(len(set(recs) & relevant) / len(relevant))
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k_frame(recommendations: pd.DataFrame, truth: pd.DataFrame, k: int = 10,
                      user_col: str = "user_id", item_col: str = "item_id") -> float:
    recs = recommendations.groupby(user_col)[item_col].apply(lambda x: list(x)[:k]).to_dict()
    targets = truth.groupby(user_col)[item_col].apply(list).to_dict()
    return recall_at_k(recs, targets, k)
