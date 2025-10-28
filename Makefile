VENV := .venv
export PATH := $(VENV)/bin:$(PATH)
PYTHON := python

regression:
	$(PYTHON) regression.py

regression-save:
	$(PYTHON) regression.py | tee outputs/regression.txt

