#!/usr/bin/env bash
set -Eeuo pipefail

# запуск из корня проекта
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- установка зависимостей ---
if [[ -f requirements.txt ]]; then
  echo '[0/4] Установка зависимостей из requirements.txt'
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
else
  echo 'Предупреждение: requirements.txt не найден, пропускаю установку зависимостей' >&2
fi
# ------------------------------

for f in data/events.csv data/user_features.csv data/item_features.csv data/submission_sample.csv role4_inputs/sasrec_scores.csv role4_inputs/sasrec_full_scores.csv; do
  if [[ ! -f "$f" ]]; then
    echo "Ошибка: отсутствует $f" >&2
    exit 1
  fi
done

mkdir -p reports candidates

echo '[1/4] Анализ данных'
"$PYTHON_BIN" -m src.eda --data-dir data --out-dir reports

echo '[2/4] Базовая валидация и рекомендации'
"$PYTHON_BIN" -m src.run_validation --data-dir data --output reports/validation_predictions.csv
"$PYTHON_BIN" -m src.make_submission --data-dir data --sample data/submission_sample.csv --output reports/submission_baseline.csv

echo '[3/4] Классические модели и кандидаты'
"$PYTHON_BIN" -m src.run_classic --submission seq_ease

echo '[4/4] Ансамбль RRF (предсказания SasRec уже включены в репозиторий)'
"$PYTHON_BIN" -m src.run_ensemble \
  --sasrec-val role4_inputs/sasrec_scores.csv \
  --sasrec-test role4_inputs/sasrec_full_scores.csv \
  --weights 'sasrec=2,seq_ease=1,ease=0.25' \
  --output submission.csv

echo 'Готово: submission.csv'
