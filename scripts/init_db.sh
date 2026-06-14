#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE airflow_metadata;
    GRANT ALL PRIVILEGES ON DATABASE airflow_metadata TO $POSTGRES_USER;
EOSQL