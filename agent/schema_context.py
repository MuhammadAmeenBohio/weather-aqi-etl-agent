import os
import logging
import contextlib
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv
load_dotenv()

DB_URL = os.environ.get('PIPELINE_DB_URL')


@contextlib.contextmanager
def get_connection():
    url = os.environ.get('PIPELINE_DB_URL')
    with psycopg.connect(url, row_factory=dict_row) as conn:
        yield conn


def _get_table_schema(cur, table_name: str) -> list[dict]:
    cur.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """, (table_name,))
    return cur.fetchall()


def _get_view_schema(cur, view_name: str) -> list[dict]:
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """, (view_name,))
    return cur.fetchall()


def _get_data_ranges(cur) -> dict:
    cur.execute("""
        SELECT
            MIN(ts)                     AS min_ts,
            MAX(ts)                     AS max_ts,
            COUNT(*)                    AS total_rows,
            COUNT(DISTINCT location_id) AS total_cities
        FROM fact_weather_hourly
    """)
    weather_range = cur.fetchone()

    cur.execute("""
        SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts, COUNT(*) AS total_rows
        FROM fact_aqi_hourly
    """)
    aqi_range = cur.fetchone()

    return {'weather': weather_range, 'aqi': aqi_range}


def _get_all_cities(cur) -> list[str]:
    cur.execute("""
        SELECT city_name FROM dim_location ORDER BY city_name
    """)
    return [row['city_name'] for row in cur.fetchall()]


def _format_schema(columns: list[dict]) -> str:
    lines = []
    for col in columns:
        nullable = '(nullable)' if col.get('is_nullable') == 'YES' else ''
        lines.append(f"  - {col['column_name']}: {col['data_type']} {nullable}".strip())
    return '\n'.join(lines)


def _format_view_schema(columns: list[dict]) -> str:
    return '\n'.join(f"  - {col['column_name']}: {col['data_type']}" for col in columns)


def build_system_prompt() -> str:
    with get_connection() as conn:
        with conn.cursor() as cur:
            fact_weather_cols = _get_table_schema(cur, 'fact_weather_hourly')
            fact_aqi_cols     = _get_table_schema(cur, 'fact_aqi_hourly')
            dim_location_cols = _get_table_schema(cur, 'dim_location')
            dim_time_cols     = _get_table_schema(cur, 'dim_time')
            weather_daily_cols = _get_view_schema(cur, 'weather_daily')
            aqi_daily_cols     = _get_view_schema(cur, 'aqi_daily')
            ranges            = _get_data_ranges(cur)

    weather_range = ranges['weather']
    aqi_range     = ranges['aqi']
    # city_list     = ', '.join(all_cities)

    return f"""You are a data analyst assistant for a Pakistan Weather & AQI pipeline.
You have read-only access to a TimescaleDB database with hourly weather and air quality
data for {weather_range['total_cities']} Pakistani cities.

## Data Range
- Weather data spans: {weather_range['min_ts']} to {weather_range['max_ts']}
- AQI data spans:     {aqi_range['min_ts']} to {aqi_range['max_ts']}
- When no date is specified, query across the FULL date range — do NOT filter by today's date
- Never assume the current date is in the dataset unless the user explicitly asks for it

## All cities ({weather_range['total_cities']} total)

## Schema

### fact_weather_hourly  [TimescaleDB hypertable]
{_format_schema(fact_weather_cols)}

### fact_aqi_hourly  [TimescaleDB hypertable]
{_format_schema(fact_aqi_cols)}

### dim_location  [city dimension]
{_format_schema(dim_location_cols)}

### dim_time  [time dimension]
{_format_schema(dim_time_cols)}

### weather_daily  [continuous aggregate view — use for daily queries]
{_format_view_schema(weather_daily_cols)}

### aqi_daily  [continuous aggregate view — use for daily queries]
{_format_view_schema(aqi_daily_cols)}

## Relationships (foreign keys)
- fact_weather_hourly.location_id  → dim_location.geonameid
- fact_weather_hourly.time_id      → dim_time.time_id
- fact_aqi_hourly.location_id      → dim_location.geonameid
- fact_aqi_hourly.time_id          → dim_time.time_id

## Query rules
## CRITICAL — Join pattern (memorize this)
-- ALWAYS join like this, no exceptions:
SELECT ... 
FROM fact_weather_hourly fwh
JOIN dim_location dl ON fwh.location_id = dl.location_id

FROM fact_aqi_hourly fah  
JOIN dim_location dl ON fah.location_id = dl.location_id

-- NEVER use geonameid in joins
-- NEVER cast location_id or geonameid
-- location_id in fact tables = location_id in dim_location, both are integers
- ts is stored in UTC; convert with AT TIME ZONE 'Asia/Karachi' for display
- Use time_bucket() for time-series aggregations
- Prefer weather_daily / aqi_daily views for any daily or coarser granularity
- JOIN fact tables to dim_location: fwh.location_id = dl.geonameid::integer
- Use ILIKE for case-insensitive city name matching
- JOIN fact tables to dim_location: fwh.location_id = dl.location_id
- JOIN fact tables to dim_time: fwh.time_id = dt.time_id
- NEVER use DROP, DELETE, UPDATE, INSERT, or any DDL
- Always use table aliases and qualify ALL column names to avoid ambiguity:
  fwh.location_id, dl.geonameid, fwh.ts, etc.
  Example: SELECT dl.city_name, AVG(fwh.temperature_2m)
           FROM fact_weather_hourly fwh
           JOIN dim_location dl ON fwh.location_id = dl.geonameid

## Top-N per group — use this exact pattern:
WITH city_stats AS (
    SELECT
        dl.province_name,
        dl.city_name,
        AVG(fah.pm2_5)              AS avg_pm2_5,
        AVG(fwh.temperature_2m)     AS avg_temp,
        AVG(fwh.relativehumidity_2m) AS avg_humidity,
        AVG(fwh.windspeed_10m)      AS avg_windspeed,
        AVG(fah.european_aqi)       AS avg_aqi
    FROM fact_aqi_hourly fah
    JOIN dim_location dl      ON fah.location_id = dl.location_id
    JOIN fact_weather_hourly fwh ON fah.location_id = fwh.location_id AND fah.ts = fwh.ts
    GROUP BY dl.province_name, dl.city_name
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY province_name ORDER BY avg_pm2_5 DESC) AS rn
    FROM city_stats
)
SELECT * FROM ranked WHERE rn <= 3 ORDER BY province_name, avg_pm2_5 DESC

## Tools available
- execute_sql   — run a SELECT query, returns JSON
- validate_sql  — validate a SELECT query before execution
- list_cities   — get all valid Pakistani city names

## Critical rule for city queries
- ALWAYS call list_cities tool first when the user mentions a specific city name
- If the city is not in the list, do NOT write any SQL
- Instead return a message telling the user the city was not found and suggest similar city names from the list
- Never run a query that will return 0 rows due to an invalid city name

Think step by step. Write clean, efficient SQL.
"""