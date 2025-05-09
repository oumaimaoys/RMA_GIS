# Final base image for Apache Airflow
FROM apache/airflow:2.10.1 as airflow_base

# Switch back to the airflow user
USER airflow

# Install Python dependencies for Airflow
COPY ./requirements.txt /
RUN pip install -r /requirements.txt

# Copy Airflow DAGs to the appropriate directory
COPY --chown=airflow:root ./dags /opt/airflow/dags

# Set the working directory for Airflow (optional)
WORKDIR /opt/airflow