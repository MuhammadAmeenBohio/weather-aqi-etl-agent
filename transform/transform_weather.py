import logging
import polars as pl
 
logger = logging.getLogger(__name__)
 
PROVINCE_MAP = {
    '02': 'Balochistan',
    '03': 'Khyber Pakhtunkhwa',
    '04': 'Punjab',
    '05': 'Sindh',
    '06': 'Azad Kashmir',
    '07': 'Gilgit-Baltistan',
    '08': 'Islamabad Capital Territory',
}

SEASON_MAP = {
    1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring',
    5: 'Summer', 6: 'Summer',
    7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
    10: 'Autumn', 11: 'Autumn',
    12: 'Winter'
}
 
REQUIRED_WEATHER_COLS = [
    'time', 'geonameid', 'city', 'province',
    'latitude', 'longitude', 'timezone', 'population',
    'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
    'apparent_temperature', 'precipitation', 'rain', 'snowfall',
    'weathercode', 'pressure_msl', 'surface_pressure', 'cloudcover',
    'visibility', 'windspeed_10m', 'winddirection_10m', 'windgusts_10m',
    'uv_index', 'is_day',
]
 
 
# ── Assertions ────────────────────────────────────────────────────────────────
 
def _assert_columns(df: pl.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'[{label}] Missing required columns: {missing}')
 
 
def _assert_no_all_null(df: pl.DataFrame, label: str) -> None:
    all_null_cols = [c for c in df.columns if df[c].is_null().all()]
    if all_null_cols:
        logger.warning(f'[{label}] All-null columns detected (API returned no data): {all_null_cols}')
 
 
def _assert_row_count(df: pl.DataFrame, label: str) -> None:
    if len(df) == 0:
        raise ValueError(f'[{label}] DataFrame has 0 rows')
 
 
# ── Transforms ────────────────────────────────────────────────────────────────
 
def _build_dim_location(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(['geonameid', 'city', 'province', 'latitude', 'longitude', 'population', 'timezone'])
        .unique(subset=['geonameid'])
        .with_columns([
            pl.col('province').replace(PROVINCE_MAP).alias('province_name'),
            pl.col('province').alias('province_code'),
        ])
        .drop('province')
        .rename({'city': 'city_name'})
    )
 
 
def _build_dim_time(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select('ts')
        .unique()
        .sort('ts')
        .with_columns([
            pl.col('ts').dt.date().alias('date'),
            pl.col('ts').dt.year().cast(pl.Int16).alias('year'),
            pl.col('ts').dt.month().cast(pl.Int16).alias('month'),
            pl.col('ts').dt.day().cast(pl.Int16).alias('day'),
            pl.col('ts').dt.hour().cast(pl.Int16).alias('hour'),
            pl.col('ts').dt.weekday().cast(pl.Int16).alias('day_of_week'),
            pl.col('ts').dt.strftime('%B').alias('month_name'),
            pl.col('ts').dt.strftime('%A').alias('day_name'),
        ])
        .with_columns([
            pl.col('day_of_week').is_in([5, 6]).alias('is_weekend'),
            pl.col('month').map_elements(
                lambda m: SEASON_MAP[m], return_dtype=pl.String
            ).alias('season'),
        ])
    )
 
 
def _build_fact_weather(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .drop(['city', 'province', 'latitude', 'longitude', 'population', 'timezone'])
        .rename({'time': 'ts'})
        .with_columns([
            pl.col('weathercode').cast(pl.Int16),
            pl.col('relativehumidity_2m').cast(pl.Int16),
            pl.col('cloudcover').cast(pl.Int16),
            pl.col('winddirection_10m').cast(pl.Int16),
            pl.col('temperature_2m').cast(pl.Float32),
            pl.col('apparent_temperature').cast(pl.Float32),
            pl.col('dewpoint_2m').cast(pl.Float32),
            pl.col('precipitation').cast(pl.Float32),
            pl.col('rain').cast(pl.Float32),
            pl.col('snowfall').cast(pl.Float32),
            pl.col('pressure_msl').cast(pl.Float32),
            pl.col('surface_pressure').cast(pl.Float32),
            pl.col('visibility').cast(pl.Float32),
            pl.col('windspeed_10m').cast(pl.Float32),
            pl.col('windgusts_10m').cast(pl.Float32),
            pl.col('uv_index').cast(pl.Float32),
            pl.col('is_day').cast(pl.Boolean),
        ])
        # Deduplicate on (ts, geonameid) before load
        .unique(subset=['ts', 'geonameid'], keep='first')
    )
 
 
# ── Main entry point ──────────────────────────────────────────────────────────
 
def run_transform_weather(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Transforms raw weather DataFrame into three DataFrames
    ready for DB load:
      - dim_location
      - dim_time
      - fact_weather_hourly
 
    Args:
        df: raw Polars DataFrame from fetch_raw_weather()
 
    Returns:
        dict with keys: 'dim_location', 'dim_time', 'fact_weather'
    """
    # ── Assertions on raw input
    _assert_columns(df, REQUIRED_WEATHER_COLS, 'raw_weather')
    _assert_row_count(df, 'raw_weather')
    _assert_no_all_null(df, 'raw_weather')
 
    logger.info(f'[transform_weather] raw input: {df.shape}')
 
    # ── Parse + localize timestamp to UTC
    df = df.with_columns(
        pl.col('time')
        .str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M')
        .dt.replace_time_zone('Asia/Karachi')
        .dt.convert_time_zone('UTC')
        .alias('time')
    )
 
    # ── Build outputs
    dim_location      = _build_dim_location(df)
    fact_weather      = _build_fact_weather(df)
    dim_time          = _build_dim_time(fact_weather)
 
    # ── Assertions on outputs
    _assert_row_count(dim_location, 'dim_location')
    _assert_row_count(dim_time,     'dim_time')
    _assert_row_count(fact_weather, 'fact_weather')
 
    logger.info(
        f'[transform_weather] '
        f'dim_location={dim_location.shape} | '
        f'dim_time={dim_time.shape} | '
        f'fact_weather={fact_weather.shape}'
    )
 
    return {
        'dim_location': dim_location,
        'dim_time':     dim_time,
        'fact_weather': fact_weather,
    }
