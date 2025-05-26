from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import logging

def trigger_django_scoring():
    url = "http://django:8000/spatial/api/run-score/"
    try:
        response = requests.post(url, json={}, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(f"❌ Status: {response.status_code}")

        raise
with DAG(
    dag_id='calculate_scores_dag',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['rma', 'scoring'],
) as dag:

    trigger_score = PythonOperator(
        task_id='trigger_score_api',
        python_callable=trigger_django_scoring,
    )
