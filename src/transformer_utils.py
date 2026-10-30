from pathlib import Path
import numpy as np
import pandas as pd


def decode_predictions(raw, encoder, history, k=100):
    users = {v: u for u, v in encoder.mapping['user_id'].items()}
    items = {v: i for i, v in encoder.mapping['item_id'].items()}
    recs = raw[['user_id', 'item_id', 'score']].copy()
    recs['user_id'] = recs['user_id'].map(users)
    recs['item_id'] = recs['item_id'].map(items)
    recs = recs.dropna()
    seen = pd.MultiIndex.from_frame(history[['user_id', 'item_id']].drop_duplicates())
    recs = recs[~pd.MultiIndex.from_frame(recs[['user_id', 'item_id']]).isin(seen)]
    recs = recs.sort_values(['user_id', 'score', 'item_id'], ascending=[True, False, True])
    return recs.drop_duplicates(['user_id', 'item_id']).groupby('user_id', sort=False).head(k)


def fill_top100(recs, history, users, catalog, k=100):
    popular = history.item_id.value_counts().index.tolist()
    candidates = list(dict.fromkeys(popular + list(catalog)))
    seen = history.groupby('user_id').item_id.agg(set).to_dict()
    predicted = {u: g[['item_id', 'score']].values.tolist() for u, g in recs.groupby('user_id')}
    result = []
    for user in users:
        used = set(seen.get(user, set()))
        selected = set()
        rows = []
        for item, score in predicted.get(user, []):
            if item not in used and item not in selected and len(rows) < k:
                rows.append((user, item, float(score)))
                selected.add(item)
        low = min((row[2] for row in rows), default=0) - 1
        for item in candidates:
            if len(rows) == k:
                break
            if item not in used and item not in selected:
                rows.append((user, item, low - len(rows) * 1e-6))
                selected.add(item)
        if len(rows) != k:
            raise ValueError(f'Недостаточно фильмов для user_id={user}')
        result.extend(rows)
    return pd.DataFrame(result, columns=['user_id', 'item_id', 'score'])


def save_submission(recs, sample, path):
    best = recs.groupby('user_id', sort=False).head(10)
    assert best.groupby('user_id').size().reindex(sample.user_id, fill_value=0).eq(10).all()
    strings = best.groupby('user_id').item_id.apply(lambda s: ' '.join(map(lambda x: str(int(x)), s)))
    sub = sample[['user_id']].merge(strings.rename('item_id'), left_on='user_id', right_index=True, how='left')
    assert sub.item_id.notna().all()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub


def rerank_genres(recs, history, item_features, weight):
    if weight == 0:
        return recs.copy()

    genres = [c for c in item_features if c.startswith('genre_')]
    liked = history.loc[history.rating >= 4, ['user_id', 'item_id']]
    profiles = liked.merge(item_features[['item_id'] + genres], on='item_id').groupby('user_id')[genres].mean()
    iv = item_features.set_index('item_id')[genres].reindex(recs.item_id).fillna(0).to_numpy(dtype=float)
    uv = profiles.reindex(recs.user_id).fillna(0).to_numpy(dtype=float)
    denominator = np.linalg.norm(iv, axis=1) * np.linalg.norm(uv, axis=1)
    sim = np.divide((iv * uv).sum(axis=1), denominator, out=np.zeros(len(recs)), where=denominator > 0)

    ranked = recs.copy()
    rank = ranked.groupby('user_id', sort=False).cumcount().to_numpy()
    ranked['raw_score'] = ranked['score']
    ranked['score'] = 1 - rank / 100 + weight * sim
    ranked = ranked.sort_values(
        ['user_id', 'score', 'raw_score', 'item_id'],
        ascending=[True, False, False, True]
    )
    return ranked.drop(columns='raw_score')


def genre_results(recs, train, truth, item_features):
    from src.metrics import recall_at_k_frame
    rows = []
    for weight in [0, 0.1, 0.25, 0.5, 1.0]:
        ranked = rerank_genres(recs, train, item_features, weight)
        recall = recall_at_k_frame(ranked.groupby('user_id', sort=False).head(10), truth, 10)
        rows.append({'genre_weight': weight, 'recall@10': recall})
    return pd.DataFrame(rows)
