CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================
-- DIMENSION TABLES
-- =============================================================

CREATE TABLE dim_location (
    location_id   SERIAL          PRIMARY KEY,
    geonameid     VARCHAR(20)     UNIQUE NOT NULL,
    city_name     VARCHAR(100)    NOT NULL,
    province_code CHAR(2),
    province_name VARCHAR(100),
    latitude      DOUBLE PRECISION NOT NULL,
    longitude     DOUBLE PRECISION NOT NULL,
    population    INTEGER,
    timezone      VARCHAR(50),
    created_at    TIMESTAMPTZ     DEFAULT NOW()
);

CREATE TABLE dim_time (
    time_id     SERIAL          PRIMARY KEY,
    ts          TIMESTAMPTZ     UNIQUE NOT NULL,
    date        DATE,
    year        SMALLINT,
    month       SMALLINT,
    month_name  VARCHAR(20),
    day         SMALLINT,
    hour        SMALLINT,
    day_of_week SMALLINT,
    day_name    VARCHAR(20),
    is_weekend  BOOLEAN,
    season      VARCHAR(20)
);

-- =============================================================
-- FACT TABLES + HYPERTABLES
-- =============================================================

CREATE TABLE fact_weather_hourly (
    ts                   TIMESTAMPTZ  NOT NULL,
    location_id          INTEGER      NOT NULL REFERENCES dim_location(location_id),
    time_id              INTEGER      REFERENCES dim_time(time_id),
    temperature_2m       REAL,
    apparent_temperature REAL,
    dewpoint_2m          REAL,
    relativehumidity_2m  SMALLINT,
    precipitation        REAL,
    rain                 REAL,
    snowfall             REAL,
    weathercode         SMALLINT,
    pressure_msl         REAL,
    surface_pressure     REAL,
    cloudcover           SMALLINT,
    visibility           REAL,
    windspeed_10m        REAL,
    winddirection_10m    SMALLINT,
    windgusts_10m        REAL,
    uv_index             REAL,
    is_day               BOOLEAN,
    PRIMARY KEY (ts, location_id)
);

SELECT create_hypertable(
    'fact_weather_hourly', 'ts',
    chunk_time_interval => INTERVAL '7 days'
);

CREATE INDEX ON fact_weather_hourly (location_id, ts DESC);
CREATE INDEX ON fact_weather_hourly (ts DESC);

-- -------------------------------------------------------------

CREATE TABLE fact_aqi_hourly (
    ts                    TIMESTAMPTZ  NOT NULL,
    location_id           INTEGER      NOT NULL REFERENCES dim_location(location_id),
    time_id               INTEGER      REFERENCES dim_time(time_id),
    pm10                  REAL,
    pm2_5                 REAL,
    carbon_monoxide       REAL,
    nitrogen_dioxide      REAL,
    sulphur_dioxide       REAL,
    ozone                 REAL,
    aerosol_optical_depth REAL,
    dust                  REAL,
    uv_index              REAL,
    european_aqi          SMALLINT,
    us_aqi                SMALLINT,
    PRIMARY KEY (ts, location_id)
);

SELECT create_hypertable(
    'fact_aqi_hourly', 'ts',
    chunk_time_interval => INTERVAL '7 days'
);

CREATE INDEX ON fact_aqi_hourly (location_id, ts DESC);
CREATE INDEX ON fact_aqi_hourly (ts DESC);

-- =============================================================
-- CONTINUOUS AGGREGATES
-- =============================================================

CREATE MATERIALIZED VIEW weather_daily
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', ts)   AS day,
        location_id,
        AVG(temperature_2m)        AS avg_temp,
        MAX(temperature_2m)        AS max_temp,
        MIN(temperature_2m)        AS min_temp,
        AVG(apparent_temperature)  AS avg_apparent_temp,
        AVG(relativehumidity_2m)   AS avg_humidity,
        SUM(precipitation)         AS total_precipitation,
        SUM(rain)                  AS total_rain,
        AVG(windspeed_10m)         AS avg_windspeed,
        MAX(windgusts_10m)         AS max_windgusts,
        AVG(cloudcover)            AS avg_cloudcover,
        MAX(uv_index)              AS max_uv_index,
        AVG(pressure_msl)          AS avg_pressure
    FROM fact_weather_hourly
    GROUP BY day, location_id
    WITH NO DATA;

SELECT add_continuous_aggregate_policy('weather_daily',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- -------------------------------------------------------------

CREATE MATERIALIZED VIEW aqi_daily
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', ts)   AS day,
        location_id,
        AVG(pm2_5)                 AS avg_pm2_5,
        MAX(pm2_5)                 AS max_pm2_5,
        AVG(pm10)                  AS avg_pm10,
        MAX(pm10)                  AS max_pm10,
        AVG(carbon_monoxide)       AS avg_co,
        AVG(nitrogen_dioxide)      AS avg_no2,
        AVG(sulphur_dioxide)       AS avg_so2,
        AVG(ozone)                 AS avg_ozone,
        AVG(dust)                  AS avg_dust,
        MAX(dust)                  AS max_dust,
        AVG(european_aqi)          AS avg_european_aqi,
        MAX(european_aqi)          AS max_european_aqi,
        AVG(us_aqi)                AS avg_us_aqi,
        MAX(us_aqi)                AS max_us_aqi
    FROM fact_aqi_hourly
    GROUP BY day, location_id
    WITH NO DATA;

SELECT add_continuous_aggregate_policy('aqi_daily',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);
