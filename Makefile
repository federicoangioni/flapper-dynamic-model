VENV := .venv
export PATH := $(VENV)/bin:$(PATH)
PYTHON := python


# Process a specific flight
regression:
	$(PYTHON) regression.py

process:
	FLIGHT=$(FLAG) $(PYTHON) process_data.py

rerun: FLAG = hover1
rerun:
	FLIGHT=$(FLAG) $(PYTHON) rerun_visuals.py