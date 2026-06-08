.PHONY: install test run airflow-init airflow-up clean

install:
	pip install -r requirements.txt

install-ci:
	pip install pandas numpy scikit-learn xgboost scipy pytest pytest-cov pyarrow joblib

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

run:
	python run_pipeline.py --data-path data/bank-additional-full.csv

run-synthetic:
	python run_pipeline.py

airflow-init:
	export AIRFLOW_HOME=$$PWD/airflow_home && airflow db init && airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

airflow-up:
	export AIRFLOW_HOME=$$PWD/airflow_home && airflow standalone

clean:
	rm -rf artifacts/ data/processed/ airflow_home/ __pycache__
	find . -name "*.pyc" -delete
