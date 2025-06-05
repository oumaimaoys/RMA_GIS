from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import logging

def trigger_django_competitors_checking():
    url = "http://django:8000/spatial/api/run-proximity-check/"
    logging.info(f"👉 POSTing to {url}")
    try:
        resp = requests.post(url, json={}, timeout=1000)
        logging.info(f"🔹 HTTP {resp.status_code}")
        logging.info(f"🔹 Response body: {resp.text!r}")
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error("❌ Request failed", exc_info=True)
        raise
with DAG(
    dag_id='competitors_check_dag',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['rma', 'scoring'],
) as dag:

    trigger_score = PythonOperator(
        task_id='trigger_score_api',
        python_callable=trigger_django_competitors_checking,
        dag=dag,
    )
    trigger_score