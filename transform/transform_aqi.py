import logging
import polars as pl
 
logger = logging.getLogger(__name__)
 
REQUIRED_AQI_COLS = [
    'time', 'geonameid', 'city', 'province',
    'latitude', 'longitude', 'timezone', 'population',
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
    'sulphur_dioxide', 'ozone', 'aerosol_optical_depth',
    'dust', 'uv_index', 'european_aqi', 'us_aqi',
]
 
 
# ── Assertions ────────────────────────────────────────────────────────────────
 
def _assert_columns(df: pl.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'[{label}] Missing required columns: {missing}')
 
 
def _assert_no_all_null(df: pl.DataFrame, label: str) -> None:
    all_null_cols = [c for c in df.columns if df[c].is_null().all()]
    if all_null_cols:
        raise ValueError(f'[{label}] All-null columns detected: {all_null_cols}')
 
 
def _assert_row_count(df: pl.DataFrame, label: str) -> None:
    if len(df) == 0:
        raise ValueError(f'[{label}] DataFrame has 0 rows')
 
 
def _assert_value_ranges(df: pl.DataFrame) -> None:
    """Sanity check on known physical bounds."""
    checks = {
        'pm2_5':         (0, 1000),
        'pm10':          (0, 2000),
        'european_aqi':  (0, 500),
        'us_aqi':        (0, 500),
    }
    for col, (lo, hi) in checks.items():
        if col not in df.columns:
            continue
        out_of_range = df.filter(
            pl.col(col).is_not_null() & ((pl.col(col) < lo) | (pl.col(col) > hi))
        )
        if len(out_of_range) > 0:
            logger.warning(
                f'[transform_aqi] {len(out_of_range)} rows outside expected range '
                f'for {col} [{lo}, {hi}] — keeping but flagging'
            )
 
 
# ── Transform ─────────────────────────────────────────────────────────────────
 
def _build_fact_aqi(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .drop(['city', 'province', 'latitude', 'longitude', 'population', 'timezone'])
        .rename({'time': 'ts'})
        .with_columns([
            # AQI indices — Int16 matches DB SMALLINT
            pl.col('european_aqi').cast(pl.Int16),
            pl.col('us_aqi').cast(pl.Int16),
            # Concentrations — Float32 matches DB REAL
            pl.col('pm10').cast(pl.Float32),
            pl.col('pm2_5').cast(pl.Float32),
            pl.col('carbon_monoxide').cast(pl.Float32),
            pl.col('nitrogen_dioxide').cast(pl.Float32),
            pl.col('sulphur_dioxide').cast(pl.Float32),
            pl.col('ozone').cast(pl.Float32),
            pl.col('aerosol_optical_depth').cast(pl.Float32),
            pl.col('dust').cast(pl.Float32),
            pl.col('uv_index').cast(pl.Float32),
        ])
        # Deduplicate on (ts, geonameid) before load
        .unique(subset=['ts', 'geonameid'], keep='first')
    )
 
 
# ── Main entry point ──────────────────────────────────────────────────────────
 
def run_transform_aqi(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Transforms raw AQI DataFrame into:
      - fact_aqi_hourly ready for DB load
 
    Note: dim_location and dim_time are shared with weather transform.
    AQI load will upsert dim_location and look up existing time_ids.
 
    Args:
        df: raw Polars DataFrame from fetch_raw_aqi()
 
    Returns:
        dict with key: 'fact_aqi'
    """
    # ── Assertions on raw input
    _assert_columns(df, REQUIRED_AQI_COLS, 'raw_aqi')
    _assert_row_count(df, 'raw_aqi')
    _assert_no_all_null(df, 'raw_aqi')
 
    logger.info(f'[transform_aqi] raw input: {df.shape}')
 
    # ── Parse + localize timestamp to UTC
    df = df.with_columns(
        pl.col('time')
        .str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M')
        .dt.replace_time_zone('Asia/Karachi')
        .dt.convert_time_zone('UTC')
        .alias('time')
    )
 
    # ── Build fact table
    fact_aqi = _build_fact_aqi(df)
 
    # ── Assertions on output
    _assert_row_count(fact_aqi, 'fact_aqi')
    _assert_value_ranges(fact_aqi)
 
    logger.info(f'[transform_aqi] fact_aqi={fact_aqi.shape}')
 
    return {'fact_aqi': fact_aqi}