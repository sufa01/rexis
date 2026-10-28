#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the repository root, regardless of the caller's current directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "== Movie recommendation pipeline =="
echo "Project: $ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Install Python 3.10+ and retry." >&2
  exit 1
fi

# Create an isolated environment if it does not exist.
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

echo "Installing/updating dependencies ..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt

# Check required files before running.
for f in data/events.csv data/user_features.csv data/item_features.csv data/submission_sample.csv; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Required file not found: $f" >&2
    echo "Place the dataset files in the repository's data/ directory." >&2
    exit 1
  fi
done

echo
echo "== 1/3: EDA =="
"$VENV_PYTHON" -m src.eda --data-dir data --out-dir reports

echo
echo "== 2/3: Validation =="
"$VENV_PYTHON" -m src.run_validation \
  --data-dir data \
  --output reports/validation_predictions.csv

echo
echo "== 3/3: Submission =="
"$VENV_PYTHON" -m src.make_submission \
  --data-dir data \
  --sample data/submission_sample.csv \
  --output submission.csv

echo
echo "Done."
echo "EDA report: reports/eda.md"
echo "Validation predictions: reports/validation_predictions.csv"
echo "Submission: submission.csv"
