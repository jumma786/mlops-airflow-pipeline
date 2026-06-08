"""
MLOps Training DAG
====================
Orchestrates the full ML training pipeline using Apache Airflow.

DAG structure:
  ingest_data → engineer_features → train_model → validate_model → notify

Schedule: Weekly (Monday 6am UTC)
Owner: jumma
"""

from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

from src.tasks.ingest   import ingest_data
from src.tasks.features import engineer_features
from src.tasks.train    import train_model
from src.tasks.validate import validate_model

# DAG defaults
DEFAULT_ARGS = {
    "owner":            "jumma",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

DATA_PATH = os.getenv("DATA_PATH", "data/bank-additional-full.csv")


def task_ingest(**context):
    meta = ingest_data(data_path=DATA_PATH)
    context["ti"].xcom_push(key="ingest_meta", value=meta)


def task_engineer(**context):
    ingest_meta = context["ti"].xcom_pull(key="ingest_meta")
    meta = engineer_features(input_path=ingest_meta["output_path"])
    context["ti"].xcom_push(key="feature_meta", value=meta)


def task_train(**context):
    feature_meta = context["ti"].xcom_pull(key="feature_meta")
    meta = train_model(input_path=feature_meta["output_path"])
    context["ti"].xcom_push(key="train_meta", value=meta)


def task_validate(**context):
    train_meta = context["ti"].xcom_pull(key="train_meta")
    meta = validate_model(train_meta=train_meta)
    context["ti"].xcom_push(key="validate_meta", value=meta)


def task_notify(**context):
    train_meta    = context["ti"].xcom_pull(key="train_meta")
    validate_meta = context["ti"].xcom_pull(key="validate_meta")
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*50}")
    print(f"  AUC:      {train_meta['auc']}")
    print(f"  CV AUC:   {train_meta['cv_auc_mean']} ± {train_meta['cv_auc_std']}")
    print(f"  Passed:   {validate_meta['passed']}")
    print(f"  Model:    {train_meta['model_path']}")
    print(f"{'='*50}")


with DAG(
    dag_id="mlops_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Weekly ML training pipeline — ingest, feature engineer, train, validate",
    schedule="0 6 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "bank-marketing"],
) as dag:

    start = EmptyOperator(task_id="start")

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest,
    )

    engineer = PythonOperator(
        task_id="engineer_features",
        python_callable=task_engineer,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
    )

    validate = PythonOperator(
        task_id="validate_model",
        python_callable=task_validate,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=task_notify,
    )

    end = EmptyOperator(task_id="end")

    start >> ingest >> engineer >> train >> validate >> notify >> end
