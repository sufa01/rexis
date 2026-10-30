#!/usr/bin/env bash
set -Eeuo pipefail

# Запуск из корня репозитория, откуда бы ни вызвали скрипт.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

# Модуль обучения SasRec (роль 2). ПРОВЕРЬТЕ имя: подставьте свой модуль из src/.
SASREC_MODULE="${SASREC_MODULE:-src.run_sasrec}"

# SKIP_SASREC=1 ./run.sh  -> пропустить долгое обучение SasRec и ансамбль
SKIP_SASREC="${SKIP_SASREC:-0}"

echo "== Movie recommendation pipeline =="
echo "Project: $ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Install Python 3.10+ and retry." >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"

echo "Installing/updating dependencies ..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt

for f in data/events.csv data/user_features.csv data/item_features.csv data/submission_sample.csv; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Required file not found: $f" >&2
    echo "Place the dataset files in the repository's data/ directory." >&2
    exit 1
  fi
done

mkdir -p reports candidates

echo
echo "== 1/5: EDA =="
"$VENV_PYTHON" -m src.eda --data-dir data --out-dir reports

echo
echo "== 2/5: Baseline validation (popular) =="
"$VENV_PYTHON" -m src.run_validation \
  --data-dir data \
  --output reports/validation_predictions.csv

echo
echo "== 3/5: Baseline submission (popular) =="
# Пишем в отдельный файл, чтобы не затереть финальный submission.csv.
"$VENV_PYTHON" -m src.make_submission \
  --data-dir data \
  --sample data/submission_sample.csv \
  --output reports/submission_baseline.csv

echo
echo "== 4/5: Classic models (popular, ItemKNN, EASE, ALS, BPR, SLIM, seq_ease) =="
# Сохраняет candidates/{val,test}_<model>.csv.gz и submission.csv по seq_ease.
"$VENV_PYTHON" -m src.run_classic --submission seq_ease

if [[ "$SKIP_SASREC" == "1" ]]; then
  echo
  echo "SKIP_SASREC=1: SasRec и ансамбль пропущены."
  echo "Final submission: submission.csv (seq_ease)"
  exit 0
fi

SASREC_FILE="${SASREC_MODULE//.//}.py"
if [[ ! -f "$SASREC_FILE" ]]; then
  echo "ERROR: $SASREC_FILE not found. Задайте SASREC_MODULE=<ваш модуль>." >&2
  exit 1
fi

echo
echo "== 5/5: SasRec + ensemble =="
# SasRec: 30 эпох, история 100, dim 128, 3 блока, 4 головы, dropout 0.3.
# Должен сохранить sasrec_scores.csv (топ-100 кандидатов со скорами).
"$VENV_PYTHON" -m "$SASREC_MODULE"

# Ансамбль SasRec + seq_ease + EASE (веса 2 : 1 : 0.25), пишет итоговый submission.csv.
"$VENV_PYTHON" -m src.run_ensemble

echo
echo "Done."
echo "EDA report:           reports/eda.md"
echo "Baseline predictions: reports/validation_predictions.csv"
echo "Baseline submission:  reports/submission_baseline.csv"
echo "Candidates:           candidates/"
echo "Final submission:     submission.csv"
