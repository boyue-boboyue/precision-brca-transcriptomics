SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := test

PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
VERIFY := $(PY) scripts/run_readonly_verifier.py --
FOREST_N_JOBS ?= 4
GDC_DOWNLOAD_WORKERS ?= 12
REBUILD_REPORTS ?= 0
export MPLCONFIGDIR := $(abspath $(VENV)/.matplotlib)

REQUIREMENTS := \
	requirements-expression.txt \
	requirements-eda.txt \
	requirements-modeling.txt \
	requirements-interpretability.txt
ENVIRONMENT_STAMP := $(VENV)/.oncostratify-environment

# These are the nine supported public entry points.  The scientific stages are
# deliberately ordered so `make test` exercises the complete delivery chain.
.PHONY: environment metadata cohort matrix eda train evaluate explain test

environment: $(ENVIRONMENT_STAMP)
	@echo "[environment] Python environment is ready: $(PY)"

$(PY):
	$(PYTHON) -m venv $(VENV)

$(ENVIRONMENT_STAMP): $(REQUIREMENTS) | $(PY)
	$(PIP) install --disable-pip-version-check -r requirements-interpretability.txt
	@touch $@

metadata: environment
	@echo "[metadata] Verifying frozen GDC query artifacts and derived metadata tables"
	$(PY) scripts/verify_source_artifacts.py --stage metadata

cohort: metadata
	@echo "[cohort] Verifying locked PAM50 cohort and source-to-label provenance"
	$(PY) scripts/verify_source_artifacts.py --stage cohort

matrix: cohort
	@echo "[matrix] Verifying matrices, or reconstructing omitted arrays from the locked GDC manifest"
	$(PY) scripts/ensure_expression_matrix.py --download-workers $(GDC_DOWNLOAD_WORKERS)

eda: matrix
	@if [[ "$(REBUILD_REPORTS)" == "1" ]]; then \
		echo "[eda] Rebuilding EDA tables and figures"; \
		$(PY) scripts/run_eda.py; \
	else \
		echo "[eda] Using frozen EDA artifacts (set REBUILD_REPORTS=1 to redraw them)"; \
	fi
	$(VERIFY) $(PY) scripts/verify_eda.py

train: eda
	@echo "[train] Verifying frozen nested-CV model artifacts"
	@echo "[train] Canonical refits are not run in place because final-test and SHAP locks hash these artifacts"
	$(VERIFY) $(PY) scripts/verify_evaluation_framework.py
	$(VERIFY) $(PY) scripts/verify_logistic_comparison.py
	$(VERIFY) $(PY) scripts/verify_linear_svc.py
	$(VERIFY) $(PY) scripts/verify_random_forest.py

evaluate: train
	@if [[ "$(REBUILD_REPORTS)" == "1" ]]; then \
		echo "[evaluate] Rebuilding the unified report from frozen OOF and one-time locked-test predictions"; \
		$(PY) scripts/generate_unified_performance_report.py; \
	else \
		echo "[evaluate] Verifying the frozen unified report; the locked test is not re-opened"; \
	fi
	$(VERIFY) $(PY) scripts/verify_unified_performance_report.py

explain: evaluate
	@if [[ "$(REBUILD_REPORTS)" == "1" ]]; then \
		echo "[explain] Rebuilding presentation artifacts from frozen explanation tables"; \
		$(PY) scripts/fetch_functional_enrichment.py --rebuild-table-from-raw; \
		$(PY) scripts/generate_interpretability_report.py --rebuild; \
		$(PY) scripts/run_robustness_analysis.py --forest-n-jobs $(FOREST_N_JOBS) --finalize-only; \
	else \
		echo "[explain] Verifying frozen explanation and robustness artifacts; SHAP test cases are not re-opened"; \
	fi
	$(VERIFY) $(PY) scripts/verify_interpretability.py
	$(VERIFY) $(PY) scripts/verify_robustness.py

test: explain
	@echo "[test] Running unit tests and all stage verifiers"
	$(PY) -m unittest discover -s tests -v
	$(PY) scripts/verify_source_artifacts.py --stage cohort
	$(VERIFY) $(PY) scripts/verify_expression_matrix.py
	$(VERIFY) $(PY) scripts/verify_eda.py
	$(VERIFY) $(PY) scripts/verify_evaluation_framework.py
	$(VERIFY) $(PY) scripts/verify_logistic_comparison.py
	$(VERIFY) $(PY) scripts/verify_linear_svc.py
	$(VERIFY) $(PY) scripts/verify_random_forest.py
	$(VERIFY) $(PY) scripts/verify_unified_performance_report.py
	$(VERIFY) $(PY) scripts/verify_interpretability.py
	$(VERIFY) $(PY) scripts/verify_robustness.py
	@echo "[test] PASS"
