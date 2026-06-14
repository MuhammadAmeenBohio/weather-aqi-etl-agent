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
 
 
def _fetch_location_id_map(cur) -> dict:
    cur.execute('SELECT geonameid, location_id FROM dim_location')
    return {row['geonameid']: row['location_id'] for row in cur.fetchall()}
 
 
def _fetch_time_id_map(cur) -> dict:
    cur.execute("SELECT ts::text, time_id FROM dim_time")
    return {row['ts']: row['time_id'] for row in cur.fetchall()}
 
 
def _insert_fact_aqi(cur, df: pl.DataFrame) -> dict:
    """
    Inserts fact_aqi_hourly rows in batches.
    Joins location_id and time_id from DB maps.
    Returns inserted and skipped counts.
    """
    loc_map  = _fetch_location_id_map(cur)
    time_map = _fetch_time_id_map(cur)
 
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
                f"[load_aqi] Skipping row — "
                f"geonameid={row['geonameid']} ts={row['ts_str']} "
                f"location_id={location_id} time_id={time_id}"
            )
            skipped += 1
            continue
 
        rows.append({
            'ts':                    row['ts_str'],
            'location_id':           location_id,
            'time_id':               time_id,
            'pm10':                  row.get('pm10'),
            'pm2_5':                 row.get('pm2_5'),
            'carbon_monoxide':       row.get('carbon_monoxide'),
            'nitrogen_dioxide':      row.get('nitrogen_dioxide'),
            'sulphur_dioxide':       row.get('sulphur_dioxide'),
            'ozone':                 row.get('ozone'),
            'aerosol_optical_depth': row.get('aerosol_optical_depth'),
            'dust':                  row.get('dust'),
            'uv_index':              row.get('uv_index'),
            'european_aqi':          row.get('european_aqi'),
            'us_aqi':                row.get('us_aqi'),
        })
 
    # Batch insert
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        cur.executemany(
            """
            INSERT INTO fact_aqi_hourly (
                ts, location_id, time_id,
                pm10, pm2_5, carbon_monoxide, nitrogen_dioxide,
                sulphur_dioxide, ozone, aerosol_optical_depth, dust,
                uv_index, european_aqi, us_aqi
            ) VALUES (
                %(ts)s, %(location_id)s, %(time_id)s,
                %(pm10)s, %(pm2_5)s, %(carbon_monoxide)s, %(nitrogen_dioxide)s,
                %(sulphur_dioxide)s, %(ozone)s, %(aerosol_optical_depth)s, %(dust)s,
                %(uv_index)s, %(european_aqi)s, %(us_aqi)s
            )
            ON CONFLICT (ts, location_id) DO NOTHING
            """,
            batch,
        )
        inserted += len(batch)
        logger.info(f'[load_aqi] batch {i // BATCH_SIZE + 1} — inserted {len(batch)} rows')
 
    return {'inserted': inserted, 'skipped': skipped}
 
 
# ── Main entry point ──────────────────────────────────────────────────────────
 
def run_load_aqi(data: dict[str, pl.DataFrame]) -> dict:
    """
    Loads fact_aqi_hourly into TimescaleDB.
    dim_location and dim_time are already populated by load_weather.
 
    Args:
        data: dict with key 'fact_aqi'
 
    Returns:
        dict with 'inserted' and 'skipped' counts
    """
    fact_aqi = data['fact_aqi']
 
    with _get_connection() as conn:
        with conn.cursor() as cur:
            result = _insert_fact_aqi(cur, fact_aqi)
        conn.commit()
 
    logger.info(
        f'[load_aqi] done — '
        f'inserted={result["inserted"]} skipped={result["skipped"]}'
    )
    return result