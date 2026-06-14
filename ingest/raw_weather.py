import os
import logging
import httpx
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
 
from ingest.cities import get_pk_cities
 
logger = logging.getLogger(__name__)
 
WEATHER_URL = os.environ.get(
    'WEATHER_API_URL',
    'https://archive-api.open-meteo.com/v1/archive'
)
 
WEATHER_FIELDS = ','.join([
    'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
    'apparent_temperature', 'precipitation', 'rain', 'snowfall',
    'weathercode', 'pressure_msl', 'surface_pressure', 'cloudcover',
    'visibility', 'windspeed_10m', 'winddirection_10m', 'windgusts_10m',
    'uv_index', 'is_day',
])
 
 
@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch_city_weather(client: httpx.Client, city: dict, start_date: str, end_date: str) -> dict:
    """
    Fetches weather for a single city. Retries up to 4x with exponential backoff.
    Raises on 429 (rate limit) or any 5xx after retries exhausted.
    """
    response = client.get(WEATHER_URL, params={
        'latitude':   city['latitude'],
        'longitude':  city['longitude'],
        'timezone':   city['timezone'],
        'start_date': start_date,
        'end_date':   end_date,
        'hourly':     WEATHER_FIELDS,
    })
 
    if response.status_code == 429:
        logger.warning(f"Rate limited on {city['city']} — will retry")
        raise httpx.HTTPStatusError(
            'Rate limited', request=response.request, response=response
        )
 
    response.raise_for_status()
    return response.json()
 
 
def fetch_raw_weather(start_date: str, end_date: str) -> pl.DataFrame:
    """
    Fetches weather for all PK cities between start_date and end_date.
    Returns a single Polars DataFrame with all cities concatenated.
 
    Args:
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'
 
    Returns:
        Polars DataFrame with columns:
            time, geonameid, city, province, latitude, longitude,
            timezone, population + all weather fields
    """
    cities = get_pk_cities()
    logger.info(f'Fetching weather for {len(cities)} cities | {start_date} → {end_date}')
 
    dfs = []
    failed = []
 
    with httpx.Client(timeout=30) as client:
        for i, city in enumerate(cities):
            try:
                raw = _fetch_city_weather(client, city, start_date, end_date)
                hourly = raw.get('hourly', {})
 
                # Guard: skip if API returned empty hourly block
                if not hourly or 'time' not in hourly:
                    logger.warning(f"[{i+1}/{len(cities)}] {city['city']} — empty response, skipping")
                    failed.append(city['city'])
                    continue
 
                # Guard: all columns must have same length as time
                time_len = len(hourly['time'])
                for col, vals in hourly.items():
                    if len(vals) != time_len:
                        raise ValueError(
                            f"Column length mismatch in {city['city']}: "
                            f"'{col}' has {len(vals)} rows, expected {time_len}"
                        )
 
                df = (
                    pl.DataFrame(hourly)
                    .with_columns([
                        pl.lit(city['geonameid']).alias('geonameid'),
                        pl.lit(city['city']).alias('city'),
                        pl.lit(city['province']).alias('province'),
                        pl.lit(city['latitude']).alias('latitude'),
                        pl.lit(city['longitude']).alias('longitude'),
                        pl.lit(city['timezone']).alias('timezone'),
                        pl.lit(city['population']).alias('population'),
                    ])
                )
                dfs.append(df)
                logger.info(f"[{i+1}/{len(cities)}] {city['city']} — {len(df)} rows OK")
 
            except Exception as e:
                logger.error(f"[{i+1}/{len(cities)}] {city['city']} FAILED — {e}")
                failed.append(city['city'])
 
    if not dfs:
        raise RuntimeError('Weather fetch returned zero dataframes — all cities failed')
 
    weather_df = pl.concat(dfs, how='diagonal')
 
    logger.info(
        f'Weather fetch complete | '
        f'success={len(dfs)} failed={len(failed)} '
        f'total_rows={len(weather_df)}'
    )
 
    if failed:
        logger.warning(f'Failed cities: {failed}')
 
    return weather_df