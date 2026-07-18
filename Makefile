# Makefile
SHELL = /bin/bash

# Quality
.PHONY: format lint test quality
format:
	black .
	python3 -m isort .
	find madewithml tests -name "*.py" -print0 | xargs -0 pyupgrade --py310-plus

lint:
	black --check .
	python3 -m isort --check-only .
	flake8

test:
	pytest tests/code tests/data --dataset-loc=datasets/dataset.csv --cov=madewithml --cov-report=term-missing

quality: lint test

# Cleaning
.PHONY: clean
clean:
	python notebooks/clear_cell_nums.py
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .coverage*
