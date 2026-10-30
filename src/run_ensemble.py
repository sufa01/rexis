"""Объединение рекомендаций нескольких моделей методом RRF.
python -m src.run_ensemble
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_candidates(path):
    frame = pd.read_csv(path)
    required = {'user_id', 'item_id'}
    if not required.issubset(frame.columns):
        raise ValueError(f'{path}: expected user_id and item_id')
    frame = frame.dropna(subset=['user_id', 'item_id']).copy()
    if 'score' not in frame.columns and 'rank' not in frame.columns:
        raise ValueError(f'{path}: need score or rank')
    if 'score' in frame:
        frame['score'] = pd.to_numeric(frame['score'], errors='coerce').fillna(-np.inf)
        frame = frame.sort_values(['user_id', 'score'], ascending=[True, False], kind='mergesort')
    else:
        frame = frame.sort_values(['user_id', 'rank'], kind='mergesort')
    frame = frame.drop_duplicates(['user_id', 'item_id'], keep='first')
    frame['rank'] = frame.groupby('user_id').cumcount() + 1
    return frame[['user_id', 'item_id', 'rank']]


def load_sources(candidate_dir, sasrec, stage, no_transformer=False):
    paths = sorted(Path(candidate_dir).glob(f'{stage}_*.csv*'))
    sources = {}
    for p in paths:
        name = p.name.replace(f'{stage}_', '').split('.csv')[0]
        if name == 'truth':
            continue
        if no_transformer and any(tag in name.lower() for tag in ('sasrec', 'bert4rec', 'transformer')):
            continue
        if name in sources:
            continue
        sources[name] = load_candidates(p)
    if not no_transformer and sasrec:
        path = Path(sasrec)
        if path.exists():
            sources['sasrec'] = load_candidates(path)
        else:
            print(f'No SasRec file at {path}; using classic models only')
    if not sources:
        raise FileNotFoundError(f'No candidate files for {stage} in {candidate_dir}')
    return sources


def combine(sources, weights, k=40):
    pieces = []
    for name, frame in sources.items():
        weight = weights.get(name, 0)
        if weight <= 0:
            continue
        piece = frame[['user_id', 'item_id', 'rank']].copy()
        piece['contribution'] = float(weight) / (k + frame['rank'].to_numpy())
        pieces.append(piece)
    if not pieces:
        raise ValueError('All model weights are zero')
    scores = (pd.concat(pieces, ignore_index=True)
              .groupby(['user_id', 'item_id'], as_index=False)
              .agg(contribution=('contribution', 'sum'), best_rank=('rank', 'min')))
    return scores.sort_values(['user_id', 'contribution', 'best_rank', 'item_id'],
                              ascending=[True, False, True, True])


def truth_map(path):
    df = pd.read_csv(path)
    if not {'user_id', 'item_id'}.issubset(df.columns):
        raise ValueError('Truth should have user_id, item_id')
    return df.groupby('user_id')['item_id'].apply(set).to_dict()


def recall_at_10(ranked, truth):
    top = ranked.groupby('user_id', sort=False).head(10)
    preds = top.groupby('user_id')['item_id'].apply(set).to_dict()
    values = [len(preds.get(user, set()) & relevant) / len(relevant)
              for user, relevant in truth.items() if relevant]
    return float(np.mean(values)) if values else float('nan')


def choose_weights(sources, truth):
    names = list(sources)
    # Проверяем небольшой набор простых весов, чтобы не переобучиться на валидации.
    variants = [{'name': 'equal', 'weights': {n: 1.0 for n in names}}]
    for lead in names:
        for boost in [2.0, 4.0]:
            variants.append({'name': f'{lead}_x{boost:g}',
                             'weights': {n: boost if n == lead else 1.0 for n in names}})
    if 'sasrec' in names and 'seq_ease' in names:
        for ratio in [0.5, 1.0, 2.0, 3.0]:
            variants.append({'name': f'sasrec_seq_{ratio:g}',
                             'weights': {n: (ratio if n == 'sasrec' else 1.0 if n == 'seq_ease' else 0.0) for n in names}})
    if 'seq_ease' in names:
        variants.append({'name': 'seq_only', 'weights': {n: float(n == 'seq_ease') for n in names}})
    if 'sasrec' in names:
        variants.append({'name': 'sasrec_only', 'weights': {n: float(n == 'sasrec') for n in names}})
    rows = []
    for option in variants:
        ranking = combine(sources, option['weights'])
        rows.append((option['name'], recall_at_10(ranking, truth), option['weights']))
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows


def write_submission(ranked, sample_path, out_path, fallback=None):
    sample = pd.read_csv(sample_path)
    if not {'user_id', 'item_id'}.issubset(sample.columns):
        raise ValueError('Sample must have user_id and item_id')
    result = ranked.groupby('user_id', sort=False)['item_id'].apply(list).to_dict()
    backup = []
    if fallback is not None:
        backup = fallback.sort_values(['user_id', 'rank']).groupby('user_id')['item_id'].apply(list).to_dict()
    global_items = ranked.groupby('item_id')['user_id'].nunique().sort_values(ascending=False).index.tolist()
    all_items = list(dict.fromkeys(global_items + [item for l in backup.values() for item in l]))
    def recommend(user):
        used = set()
        selected = []
        for item in result.get(user, [])[:100] + backup.get(user, []) + all_items:
            if item not in used:
                selected.append(item)
                used.add(item)
            if len(selected) == 10:
                break
        if len(selected) != 10:
            raise ValueError(f'Only {len(selected)} candidates for user {user}')
        return ' '.join(str(int(x)) if isinstance(x, (np.integer, int, float)) and float(x).is_integer() else str(x) for x in selected)
    sample['item_id'] = sample['user_id'].map(recommend)
    if sample['item_id'].isna().any():
        raise ValueError('Missing predictions for some users')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f'Saved {len(sample)} users to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-dir', default='candidates')
    parser.add_argument('--sasrec-val', default='role4_inputs/sasrec_scores.csv')
    parser.add_argument('--sasrec-test', default='role4_inputs/sasrec_full_scores.csv')
    parser.add_argument('--sample', default='data/submission_sample.csv')
    parser.add_argument('--output', default='submission_ensemble.csv')
    parser.add_argument('--no-transformer', action='store_true', help='For a transformer-free final solution')
    parser.add_argument('--weights', default='', help='e.g. sasrec=2,seq_ease=1; skips weight tuning')
    args = parser.parse_args()
    if not Path(args.sample).exists() and Path('submission_sample.csv').exists():
        args.sample = 'submission_sample.csv'
    val = load_sources(args.candidate_dir, args.sasrec_val, 'val', args.no_transformer)
    test = load_sources(args.candidate_dir, args.sasrec_test, 'test', args.no_transformer)
    common = sorted(set(val) & set(test))
    if not common:
        raise ValueError('No models found in both val and test')
    val, test = ({name: sources[name] for name in common} for sources in [val, test])
    print('Models:', ', '.join(common))
    if args.weights:
        weights = {name: float(value) for name, value in (part.split('=', 1) for part in args.weights.split(','))}
        unknown = set(weights) - set(common)
        if unknown:
            raise ValueError(f'Weights reference missing models: {sorted(unknown)}')
        if not any(v > 0 for v in weights.values()):
            raise ValueError('At least one model must have positive weight')
        print('Manual weights:', weights)
    else:
        path = Path(args.candidate_dir) / 'val_truth.csv.gz'
        if path.exists():
            results = choose_weights(val, truth_map(path))
            print('Validation results (R@10):')
            for name, score, _ in results:
                print(f'  {name:22s} {score:.6f}')
            weights = results[0][2]
            print('Best:', results[0][0], weights)
        else:
            print('No validation truth; using equal weights (no metric available)')
            weights = {name: 1.0 for name in common}
    # Сохраняем веса, чтобы можно было воспроизвести результат.
    import json
    Path('reports').mkdir(exist_ok=True)
    with open('reports/ensemble_weights.json', 'w', encoding='utf-8') as fp:
        json.dump(weights, fp, indent=2)
    ranked = combine(test, weights)
    fallback = test.get('seq_ease', next(iter(test.values())))
    write_submission(ranked, args.sample, args.output, fallback)


if __name__ == '__main__':
    main()
