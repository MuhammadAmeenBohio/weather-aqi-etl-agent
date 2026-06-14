import logging
import io
from datetime import datetime, timedelta
 
import polars as pl
from airflow.decorators import dag, task
import base64
import io
 
logger = logging.getLogger(__name__)
 
# ── Serialization helpers ─────────────────────────────────────────────────────
 
def _serialize(data: dict[str, pl.DataFrame]) -> dict[str, str]:
    """Serialize dict of DataFrames to dict of base64 strings for XCom."""
    return {k: base64.b64encode(v.write_ipc(None).getvalue()).decode('utf-8') for k, v in data.items()}

def _deserialize(data: dict[str, str]) -> dict[str, pl.DataFrame]:
    """Deserialize dict of base64 strings back to dict of DataFrames."""
    return {k: pl.read_ipc(io.BytesIO(base64.b64decode(v))) for k, v in data.items()}
 
 
# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    'owner':                     'pk_weather_pipeline',
    'retries':                   2,
    'retry_delay':               timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'execution_timeout':         timedelta(hours=2),
}
 
 
# ── DAG ───────────────────────────────────────────────────────────────────────
@dag(
    dag_id='weather_aqi_pipeline',
    description='Hourly weather + AQI ETL for all Pakistan cities',
    schedule='0 0 * * *',          # 00:00 UTC = 05:00 PKT
    start_date=datetime(2025, 1, 1),
    catchup=False,                  # set True for historical backfill
    max_active_runs=1,
    default_args=default_args,
    tags=['weather', 'aqi', 'pakistan'],
)
def weather_aqi_pipeline():
 
    # ── 1. Ingest ─────────────────────────────────────────────────────────────
 
    @task()
    def fetch_weather(logical_date=None) -> bytes:
        from ingest.raw_weather import fetch_raw_weather
 
        start_date = logical_date.strftime('%Y-%m-%d')
        logger.info(f'[fetch_weather] date={start_date}')
 
        df = fetch_raw_weather(start_date, start_date)
        logger.info(f'[fetch_weather] rows={len(df)}')
 
        return base64.b64encode(df.write_ipc(None).getvalue()).decode('utf-8')
    
 
 
    @task()
    def fetch_aqi(logical_date=None) -> bytes:
        from ingest.raw_aqi import fetch_raw_aqi
 
        start_date = logical_date.strftime('%Y-%m-%d')
        logger.info(f'[fetch_aqi] date={start_date}')
 
        df = fetch_raw_aqi(start_date, start_date)
        logger.info(f'[fetch_aqi] rows={len(df)}')
 
        return base64.b64encode(df.write_ipc(None).getvalue()).decode('utf-8')
 
 
    # ── 2. Transform ──────────────────────────────────────────────────────────
 
    @task()
    def transform_weather(raw_bytes: bytes) -> dict[str, bytes]:
        """
        Returns serialized dict:
          { 'dim_location': bytes, 'dim_time': bytes, 'fact_weather': bytes }
        """
        from transform.transform_weather import run_transform_weather
 
        df = pl.read_ipc(io.BytesIO(base64.b64decode(raw_bytes)))
        logger.info(f'[transform_weather] input rows={len(df)}')
 
        result = run_transform_weather(df)
        logger.info(
            f'[transform_weather] '
            f'dim_location={result["dim_location"].shape} | '
            f'dim_time={result["dim_time"].shape} | '
            f'fact_weather={result["fact_weather"].shape}'
        )
 
        return _serialize(result)
 
 
    @task()
    def transform_aqi(raw_bytes: bytes) -> dict[str, bytes]:
        """
        Returns serialized dict:
          { 'fact_aqi': bytes }
        """
        from transform.transform_aqi import run_transform_aqi
 
        df = pl.read_ipc(io.BytesIO(base64.b64decode(raw_bytes)))
        logger.info(f'[transform_aqi] input rows={len(df)}')
 
        result = run_transform_aqi(df)
        logger.info(f'[transform_aqi] fact_aqi={result["fact_aqi"].shape}')
 
        return _serialize(result)
 
 
    # ── 3. Load ───────────────────────────────────────────────────────────────
 
    @task()
    def load_weather(serialized: dict[str, bytes]) -> dict:
        """
        Loads dim_location, dim_time, fact_weather_hourly.
        Must run before load_aqi so dim tables are populated.
        """
        from load.load_weather import run_load_weather
 
        data = _deserialize(serialized)
        logger.info(f'[load_weather] fact rows={len(data["fact_weather"])}')
 
        result = run_load_weather(data)
        logger.info(f'[load_weather] inserted={result["inserted"]} skipped={result["skipped"]}')
 
        return result
 
 
    @task()
    def load_aqi(serialized: dict[str, bytes], weather_result: dict) -> dict:
        """
        Loads fact_aqi_hourly.
        weather_result dependency ensures dim tables exist before this runs.
        """
        from load.load_aqi import run_load_aqi
 
        data = _deserialize(serialized)
        logger.info(f'[load_aqi] fact rows={len(data["fact_aqi"])}')
 
        result = run_load_aqi(data)
        logger.info(f'[load_aqi] inserted={result["inserted"]} skipped={result["skipped"]}')
 
        return result
 
 
    # ── 4. Verify ─────────────────────────────────────────────────────────────
 
    @task()
    def verify_load(weather_result: dict, aqi_result: dict, logical_date=None) -> None:
        """
        Fails loudly if either load inserted 0 rows.
        Logs final counts for Airflow task logs.
        """
        date_str = logical_date.strftime('%Y-%m-%d')
 
        logger.info(
            f'[verify_load] date={date_str} | '
            f'weather inserted={weather_result["inserted"]} | '
            f'aqi inserted={aqi_result["inserted"]}'
        )
 
        if weather_result['inserted'] == 0:
            raise ValueError(f'Weather load inserted 0 rows for {date_str}')
        if aqi_result['inserted'] == 0:
            raise ValueError(f'AQI load inserted 0 rows for {date_str}')
 
 
    # ── Wire up the graph ─────────────────────────────────────────────────────
    #
    #   fetch_weather ──► transform_weather ──► load_weather ──────────────────┐
    #                                                    │                      ├──► verify_load
    #   fetch_aqi     ──► transform_aqi     ──► load_aqi(needs weather_result)─┘
    #
 
    raw_weather_bytes   = fetch_weather()
    raw_aqi_bytes       = fetch_aqi()
 
    transformed_weather = transform_weather(raw_weather_bytes)
    transformed_aqi     = transform_aqi(raw_aqi_bytes)
 
    weather_result      = load_weather(transformed_weather)
    aqi_result          = load_aqi(transformed_aqi, weather_result)  # explicit dependency
 
    verify_load(weather_result, aqi_result)
 
 
# Instantiate
weather_aqi_pipeline()