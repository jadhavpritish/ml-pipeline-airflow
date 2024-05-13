#!/bin/bash

APP_HOME=$(pwd)

AIRFLOW_HOME=${APP_HOME} poetry run airflow db init
AIRFLOW_HOME=${APP_HOME} poetry run airflow users create --username admin --password admin --firstname Anonymous --lastname Admin --role Admin --email admin@example.org
AIRFLOW_HOME=${APP_HOME} poetry run airflow scheduler &
AIRFLOW_HOME=${APP_HOME} poetry run airflow webserver