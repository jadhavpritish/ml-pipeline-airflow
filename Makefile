SHELL := /bin/bash

.PHONY = init clean

AIRFLOW_HOME ?= $$(pwd)
AIRFLOW_DAGS_DIRECTORY ?= $$(pwd)/dags

clean: ## Clean

	# git gc is really just minor tidying - https://git-scm.com/docs/git-gc
	git gc --aggressive


init: clean
	poetry run pip install --upgrade pip

	@# If you are having problems running this on Big Sur, try prepending the following:  SYSTEM_VERSION_COMPAT=1
	poetry install

format:
	@echo "AIRFLOW_HOME = ${AIRFLOW_HOME}"
	@echo "AIRFLOW_DAGS_DIRECTORY = ${AIRFLOW_DAGS_DIRECTORY}"
	poetry run isort .
	poetry run black .


run-local:
	sh ./startup.sh

kill-local:
	pkill airflow