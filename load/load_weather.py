import os
import logging
import psycopg
from psycopg.rows import dict_row
import polars as pl
 
logger = logging.getLogger(__name__)
 
DB_URL = os.environ.get('PIPELINE_DB_URL')
BATCH_SIZE = 500
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _get_connection():
    return psycopg.connect(DB_URL, row_factory=dict_row)
 
 
def _upsert_dim_location(cur, df: pl.DataFrame) -> None:
    """Insert new locations, skip existing ones."""
    rows = df.to_dicts()
    cur.executemany(
        """
        INSERT INTO dim_location
            (geonameid, city_name, province_code, province_name,
             latitude, longitude, population, timezone)
        VALUES
            (%(geonameid)s, %(city_name)s, %(province_code)s, %(province_name)s,
             %(latitude)s, %(longitude)s, %(population)s, %(timezone)s)
        ON CONFLICT (geonameid) DO NOTHING
        """,
        rows,
    )
    logger.info(f'[load_weather] dim_location upserted {len(rows)} rows')
 
 
def _upsert_dim_time(cur, df: pl.DataFrame) -> None:
    """Insert new time entries, skip existing ones."""
    rows = df.with_columns(
        pl.col('ts').dt.strftime('%Y-%m-%dT%H:%M:%S%z').alias('ts')
    ).to_dicts()
 
    cur.executemany(
        """
        INSERT INTO dim_time
            (ts, date, year, month, month_name, day, hour,
             day_of_week, day_name, is_weekend, season)
        VALUES
            (%(ts)s, %(date)s, %(year)s, %(month)s, %(month_name)s,
             %(day)s, %(hour)s, %(day_of_week)s, %(day_name)s,
             %(is_weekend)s, %(season)s)
        ON CONFLICT (ts) DO NOTHING
        """,
        rows,
    )
    logger.info(f'[load_weather] dim_time upserted {len(rows)} rows')
 
 
def _fetch_location_id_map(cur) -> dict:
    """Returns {geonameid: location_id} for all locations in DB."""
    cur.execute('SELECT geonameid, location_id FROM dim_location')
    return {row['geonameid']: row['location_id'] for row in cur.fetchall()}
 
 
def _fetch_time_id_map(cur) -> dict:
    """Returns {ts_str: time_id} for all time entries in DB."""
    cur.execute("SELECT ts::text, time_id FROM dim_time")
    return {row['ts']: row['time_id'] for row in cur.fetchall()}
 
 
def _insert_fact_weather(cur, df: pl.DataFrame) -> dict:
    """
    Inserts fact_weather_hourly rows in batches.
    Joins location_id and time_id from DB maps.
    Returns inserted and skipped counts.
    """
    loc_map  = _fetch_location_id_map(cur)
    time_map = _fetch_time_id_map(cur)
 
    # Attach location_id and time_id
    df = df.with_columns(
        pl.col('ts').dt.strftime('%Y-%m-%d %H:%M:%S+00').alias('ts_str')
    )
 
    rows = []
    skipped = 0
    for row in df.to_dicts():
        location_id = loc_map.get(row['geonameid'])
        time_id     = time_map.get(row['ts_str'])
 
        if location_id is None or time_id is None:
            logger.warning(
                f"[load_weather] Skipping row — "
                f"geonameid={row['geonameid']} ts={row['ts_str']} "
                f"location_id={location_id} time_id={time_id}"
            )
            skipped += 1
            continue
 
        rows.append({
            'ts':                   row['ts_str'],
            'location_id':          location_id,
            'time_id':              time_id,
            'temperature_2m':       row.get('temperature_2m'),
            'apparent_temperature': row.get('apparent_temperature'),
            'dewpoint_2m':          row.get('dewpoint_2m'),
            'relativehumidity_2m':  row.get('relativehumidity_2m'),
            'precipitation':        row.get('precipitation'),
            'rain':                 row.get('rain'),
            'snowfall':             row.get('snowfall'),
            'weathercode':          row.get('weathercode'),
            'pressure_msl':         row.get('pressure_msl'),
            'surface_pressure':     row.get('surface_pressure'),
            'cloudcover':           row.get('cloudcover'),
            'visibility':           row.get('visibility'),
            'windspeed_10m':        row.get('windspeed_10m'),
            'winddirection_10m':    row.get('winddirection_10m'),
            'windgusts_10m':        row.get('windgusts_10m'),
            'uv_index':             row.get('uv_index'),
            'is_day':               row.get('is_day'),
        })
 
    # Batch insert
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        cur.executemany(
            """
            INSERT INTO fact_weather_hourly (
                ts, location_id, time_id,
                temperature_2m, apparent_temperature, dewpoint_2m,
                relativehumidity_2m, precipitation, rain, snowfall,
                weathercode, pressure_msl, surface_pressure, cloudcover,
                visibility, windspeed_10m, winddirection_10m, windgusts_10m,
                uv_index, is_day
            ) VALUES (
                %(ts)s, %(location_id)s, %(time_id)s,
                %(temperature_2m)s, %(apparent_temperature)s, %(dewpoint_2m)s,
                %(relativehumidity_2m)s, %(precipitation)s, %(rain)s, %(snowfall)s,
                %(weathercode)s, %(pressure_msl)s, %(surface_pressure)s, %(cloudcover)s,
                %(visibility)s, %(windspeed_10m)s, %(winddirection_10m)s, %(windgusts_10m)s,
                %(uv_index)s, %(is_day)s
            )
            ON CONFLICT (ts, location_id) DO NOTHING
            """,
            batch,
        )
        inserted += len(batch)
        logger.info(f'[load_weather] batch {i // BATCH_SIZE + 1} — inserted {len(batch)} rows')
 
    return {'inserted': inserted, 'skipped': skipped}
 
 
# ── Main entry point ──────────────────────────────────────────────────────────
 
def run_load_weather(data: dict[str, pl.DataFrame]) -> dict:
    """
    Loads all three DataFrames into TimescaleDB in a single transaction.
 
    Args:
        data: dict with keys 'dim_location', 'dim_time', 'fact_weather'
 
    Returns:
        dict with 'inserted' and 'skipped' counts
    """
    dim_location = data['dim_location']
    dim_time     = data['dim_time']
    fact_weather = data['fact_weather']
 
    with _get_connection() as conn:
        with conn.cursor() as cur:
            _upsert_dim_location(cur, dim_location)
            _upsert_dim_time(cur, dim_time)
            result = _insert_fact_weather(cur, fact_weather)
        conn.commit()
 
    logger.info(
        f'[load_weather] done — '
        f'inserted={result["inserted"]} skipped={result["skipped"]}'
    )
    return result