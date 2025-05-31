# dags/calculate_scores_dag.py

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# ── CONFIG ────────────────────────────────────────────────────────────────
# Make sure your Airflow scheduler container has access to the Docker socket:
#   volumes:
#     - /var/run/docker.sock:/var/run/docker.sock
#
# In docker-compose, the Django service is named "django" (change if yours is different).
DJANGO_SERVICE_NAME = "django"
DJANGO_WORKDIR      = "/app"    # where your manage.py lives inside the Django container

# ── DAG DEFINITION ────────────────────────────────────────────────────────
default_args = {
    "owner": "rma",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
}

with DAG(
    dag_id="calculate_scores_bash",
    default_args=default_args,
    schedule_interval=None,   # or "0 2 * * *" if you want a daily run at 02:00
    catchup=False,
    tags=["rma", "scoring"],
) as dag:

    run_calculate_scores = BashOperator(
        task_id="run_django_calculate_scores",
        bash_command=(
            # 1) Use `docker exec` to run inside the existing Django container.
            # 2) --workdir ensures we start in /app so manage.py is on PATH.
            f"docker exec "
            f"--workdir {DJANGO_WORKDIR} "
            f"{DJANGO_SERVICE_NAME} "
            f"python manage.py calculate_scores --parallel"
        ),
    )

    run_calculate_scores
