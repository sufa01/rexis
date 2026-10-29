# Movie Recommendation System

Воспроизводимый baseline для рекомендации 10 фильмов каждому пользователю по истории оценок. Метрика — Recall@10.

## Данные

В `data/` находятся:
- `events.csv`: `user_id`, `item_id`, `rating`, `timestamp`;
- `user_features.csv`: `user_id`, `gender`, `age`;
- `item_features.csv`: `item_id`, бинарные признаки `genre_0`–`genre_17`;
- `submission_sample.csv`: `user_id`, `item_id`, где `item_id` — строка из 10 ID через пробел.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск всего pipeline

```bash
make run
```

Создаёт `reports/eda.md`, рассчитывает Recall@10 на leave-last-out валидации, сохраняет `reports/validation_predictions.csv` и генерирует `submission.csv`.

## Отдельные команды

```bash
python -m src.eda --data-dir data --out-dir reports
python -m src.run_validation --data-dir data
python -m src.run_validation --positive-mode threshold --threshold 4
python -m src.make_submission --data-dir data --sample data/submission_sample.csv --output submission.csv
```

## Валидация и baseline

Базовая модель ранжирует фильмы по количеству взаимодействий. В валидации последнее уникальное взаимодействие пользователя удерживается как целевое, остальные события используются для построения рекомендаций. Уже просмотренные фильмы исключаются. Параметр `--positive-mode threshold` позволяет проверять вариант, где позитивом считается оценка не ниже заданного порога.

В данных `timestamp` — целочисленный порядок событий, а не календарная дата. Если нужно учитывать давность, `--half-life` работает в единицах этого числового столбца.

## Формат сабмита

Одна строка на пользователя; `item_id` содержит ровно 10 идентификаторов фильмов через пробел, например:

```csv
user_id,item_id
0,101 204 315 404 512 623 700 801 900 1001
```

Проверьте перед отправкой, что у каждой строки ровно 10 уникальных item ID и что формат соответствует sample.

## Эксперименты

Все гипотезы и фактические метрики записывайте в `EXPERIMENTS.md`. Результаты лидерборда нельзя подставлять до реальной отправки сабмита.

## Модели без трансформеров (роль 3)

```bash
python -m src.run_classic --submission seq_ease
```

Модели: popular, ItemKNN, EASE, ALS, BPR, SLIM, EASE + контекст дыр (`ease_gap`), EASE со свежестью + переходы (`seq_ease`, лучшая). Скрипт:
- валидирует каждую модель на `tail_holdout` (см. `src/split.py`: у каждого пользователя откладываются последние n/9 событий; это совпадает с лидербордом);
- сохраняет топ-100 кандидатов в `candidates/val_<model>.csv.gz` и `candidates/test_<model>.csv.gz`;
- пишет `submission.csv` по выбранной модели.

Весь прогон ~2.5 минуты на CPU. Результаты описаны в `EXPERIMENTS.md`.
