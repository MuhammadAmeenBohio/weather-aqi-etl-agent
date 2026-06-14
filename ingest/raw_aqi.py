import os
import logging
import httpx
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
 
from ingest.cities import get_pk_cities
 
logger = logging.getLogger(__name__)
 
AQI_URL = os.environ.get(
    'AQI_API_URL',
    'https://air-quality-api.open-meteo.com/v1/air-quality'
)
 
AQI_FIELDS = ','.join([
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
    'sulphur_dioxide', 'ozone', 'aerosol_optical_depth',
    'dust', 'uv_index', 'european_aqi', 'us_aqi',
])
 
 
@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch_city_aqi(client: httpx.Client, city: dict, start_date: str, end_date: str) -> dict:
    """
    Fetches AQI for a single city. Retries up to 4x with exponential backoff.
    """
    response = client.get(AQI_URL, params={
        'latitude':   city['latitude'],
        'longitude':  city['longitude'],
        'timezone':   city['timezone'],
        'start_date': start_date,
        'end_date':   end_date,
        'hourly':     AQI_FIELDS,
    })
 
    if response.status_code == 429:
        logger.warning(f"Rate limited on {city['city']} — will retry")
        raise httpx.HTTPStatusError(
            'Rate limited', request=response.request, response=response
        )
 
    response.raise_for_status()
    return response.json()
 
 
def fetch_raw_aqi(start_date: str, end_date: str) -> pl.DataFrame:
    """
    Fetches AQI for all PK cities between start_date and end_date.
    Returns a single Polars DataFrame with all cities concatenated.
 
    Args:
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'
 
    Returns:
        Polars DataFrame with columns:
            time, geonameid, city, province, latitude, longitude,
            timezone, population + all AQI fields
    """
    cities = get_pk_cities()
    logger.info(f'Fetching AQI for {len(cities)} cities | {start_date} → {end_date}')
 
    dfs = []
    failed = []
 
    with httpx.Client(timeout=30) as client:
        for i, city in enumerate(cities):
            try:
                raw = _fetch_city_aqi(client, city, start_date, end_date)
                hourly = raw.get('hourly', {})
 
                # Guard: skip if API returned empty hourly block
                if not hourly or 'time' not in hourly:
                    logger.warning(f"[{i+1}/{len(cities)}] {city['city']} — empty response, skipping")
                    failed.append(city['city'])
                    continue
 
                # Guard: column length consistency
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
        raise RuntimeError('AQI fetch returned zero dataframes — all cities failed')
 
    aqi_df = pl.concat(dfs, how='diagonal')
 
    logger.info(
        f'AQI fetch complete | '
        f'success={len(dfs)} failed={len(failed)} '
        f'total_rows={len(aqi_df)}'
    )
 
    if failed:
        logger.warning(f'Failed cities: {failed}')
 
    return aqi_df