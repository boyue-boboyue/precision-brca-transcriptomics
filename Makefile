SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := test

PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
VERIFY := $(PY) scripts/run_readonly_verifier.py --
N_JOBS ?= 1
FOREST_N_JOBS ?= 4
GDC_DOWNLOAD_WORKERS ?= 12
REBUILD_REPORTS ?= 0
REPRODUCTION ?= 0
REPRO_DIR ?= work/reproduction/default
REPRO_DRY_RUN ?= 0
REPRO_CMD = $(PY) scripts/run_reproduction.py \
	--workspace "$(REPRO_DIR)" \
	--n-jobs $(N_JOBS) \
	--forest-n-jobs $(FOREST_N_JOBS) \
	--download-workers $(GDC_DOWNLOAD_WORKERS) \
	$(if $(filter 1,$(REPRO_DRY_RUN)),--dry-run,)
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
	@if [[ "$(REPRODUCTION)" == "1" ]]; then \
		echo "[metadata] Reproduction mode: isolated source verification"; \
		$(REPRO_CMD) metadata; \
	else \
		echo "[metadata] Verifying frozen GDC query artifacts and derived metadata tables"; \
		$(PY) scripts/verify_source_artifacts.py --stage metadata; \
	fi

cohort: metadata
	@if [[ "$(REPRODUCTION)" == "1" ]]; then \
		echo "[cohort] Reproduction mode: isolated PAM50 cohort verification"; \
		$(REPRO_CMD) cohort; \
	else \
		echo "[cohort] Verifying locked PAM50 cohort and source-to-label provenance"; \
		$(PY) scripts/verify_source_artifacts.py --stage cohort \
			--skip-unavailable-publication-supplement; \
	fi

matrix: cohort
	@if [[ "$(REPRODUCTION)" == "1" ]]; then \
		echo "[matrix] Reproduction mode: isolated matrix verification or reconstruction"; \
		$(REPRO_CMD) matrix; \
	else \
		echo "[matrix] Verifying matrices, or reconstructing omitted arrays from the locked GDC manifest"; \
		$(PY) scripts/ensure_expression_matrix.py --download-workers $(GDC_DOWNLOAD_WORKERS); \
	fi

eda: matrix
	@if [[ "$(REPRODUCTION)" == "1" ]]; then \
		echo "[eda] Reproduction mode: rebuilding EDA in $(REPRO_DIR)"; \
		$(REPRO_CMD) eda; \
	elif [[ "$(REBUILD_REPORTS)" == "1" ]]; then \
		echo "[eda] Rebuilding EDA tables and figures"; \
		$(PY) scripts/run_eda.py; \
		$(VERIFY) $(PY) scripts/verify_eda.py; \
	else \
		echo "[eda] Using frozen EDA artifacts (set REBUILD_REPORTS=1 to redraw them)"; \
		$(VERIFY) $(PY) scripts/verify_eda.py; \
	fi

train: eda
	@if [[ "$(REPRODUCTION)" == "1" ]]; then \
		echo "[train] Reproduction mode: refitting all development-only models in $(REPRO_DIR)"; \
		$(REPRO_CMD) train; \
	else \
		echo "[train] Verifying frozen nested-CV model artifacts"; \
		echo "[train] Canonical refits are not run in place because final-test and SHAP locks hash these artifacts"; \
		$(VERIFY) $(PY) scripts/verify_evaluation_framework.py; \
		$(VERIFY) $(PY) scripts/verify_logistic_comparison.py; \
		$(VERIFY) $(PY) scripts/verify_linear_svc.py; \
		$(VERIFY) $(PY) scripts/verify_random_forest.py; \
	fi

evaluate: train
	@if [[ "$(REPRODUCTION)" == "1" ]]; then \
		echo "[evaluate] Reproduction mode: creating independent final lock and evaluation record"; \
		$(REPRO_CMD) evaluate; \
	elif [[ "$(REBUILD_REPORTS)" == "1" ]]; then \
		echo "[evaluate] Rebuilding the unified report from frozen OOF and one-time locked-test predictions"; \
		$(PY) scripts/generate_unified_performance_report.py; \
		$(VERIFY) $(PY) scripts/verify_unified_performance_report.py; \
	else \
		echo "[evaluate] Verifying the frozen unified report; the locked test is not re-opened"; \
		$(VERIFY) $(PY) scripts/verify_unified_performance_report.py; \
	fi

explain: evaluate
	@if [[ "$(REPRODUCTION)" == "1" ]]; then \
		echo "[explain] Reproduction mode: replaying explanations and robustness analyses"; \
		$(REPRO_CMD) explain; \
	elif [[ "$(REBUILD_REPORTS)" == "1" ]]; then \
		echo "[explain] Rebuilding presentation artifacts from frozen explanation tables"; \
		$(PY) scripts/fetch_functional_enrichment.py --rebuild-table-from-raw; \
		$(PY) scripts/generate_interpretability_report.py --rebuild; \
		$(PY) scripts/run_robustness_analysis.py --forest-n-jobs $(FOREST_N_JOBS) --finalize-only; \
		$(VERIFY) $(PY) scripts/verify_interpretability.py; \
		$(VERIFY) $(PY) scripts/verify_robustness.py; \
	else \
		echo "[explain] Verifying frozen explanation and robustness artifacts; SHAP test cases are not re-opened"; \
		$(VERIFY) $(PY) scripts/verify_interpretability.py; \
		$(VERIFY) $(PY) scripts/verify_robustness.py; \
	fi

test: explain
	@if [[ "$(REPRODUCTION)" == "1" ]]; then \
		echo "[test] Reproduction mode: verifying isolated replay and canonical equivalence"; \
		$(REPRO_CMD) test; \
	else \
		echo "[test] Running unit tests and all stage verifiers"; \
		$(PY) -m unittest discover -s tests -v; \
		$(PY) scripts/verify_source_artifacts.py --stage cohort \
			--skip-unavailable-publication-supplement; \
		$(VERIFY) $(PY) scripts/verify_expression_matrix.py; \
		$(VERIFY) $(PY) scripts/verify_eda.py; \
		$(VERIFY) $(PY) scripts/verify_evaluation_framework.py; \
		$(VERIFY) $(PY) scripts/verify_logistic_comparison.py; \
		$(VERIFY) $(PY) scripts/verify_linear_svc.py; \
		$(VERIFY) $(PY) scripts/verify_random_forest.py; \
		$(VERIFY) $(PY) scripts/verify_unified_performance_report.py; \
		$(VERIFY) $(PY) scripts/verify_interpretability.py; \
		$(VERIFY) $(PY) scripts/verify_robustness.py; \
		echo "[test] PASS"; \
	fi
