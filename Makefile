PYTHON ?= python

.PHONY: run eda validate submission

run: eda validate submission

eda:
	$(PYTHON) -m src.eda --data-dir data --out-dir reports

validate:
	$(PYTHON) -m src.run_validation --data-dir data --output reports/validation_predictions.csv

submission:
	$(PYTHON) -m src.make_submission --data-dir data --sample data/submission_sample.csv --output submission.csv
