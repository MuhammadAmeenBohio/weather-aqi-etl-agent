"""
Data Quality Diagnostic Script
Run this from the project root to diagnose province mapping
and duplicate issues in the pipeline data.
"""

import json
import polars as pl
import geonamescache
from pathlib import Path

gc = geonamescache.GeonamesCache()
all_cities = gc.get_cities()

# ── 1. Check actual province codes ────────────────────────────────────────────
print("=" * 60)
print("1. UNIQUE PROVINCE CODES IN GEONAMESCACHE FOR PAKISTAN")
print("=" * 60)

pk_cities = [
    {
        'geonameid':  gid,
        'city':       city['name'],
        'admin1code': city['admin1code'],
        'timezone':   city['timezone'],
    }
    for gid, city in all_cities.items()
    if city['countrycode'] == 'PK'
]

unique_codes = sorted(set(c['admin1code'] for c in pk_cities))
print(f"Unique admin1codes: {unique_codes}")
print()

# Show a few cities per code so we can identify the province
print("Sample cities per admin1code:")
from collections import defaultdict
code_to_cities = defaultdict(list)
for c in pk_cities:
    code_to_cities[c['admin1code']].append(c['city'])

for code in sorted(code_to_cities):
    sample = code_to_cities[code][:3]
    print(f"  {code!r:6} → {sample}")

# ── 2. Check for duplicate (ts, location_id) in fact tables ──────────────────
print()
print("=" * 60)
print("2. DUPLICATE CHECK IN FACT TABLES")
print("=" * 60)

RAW_WEATHER = Path('raw/weather_pakistan.json')
RAW_AQI     = Path('raw/aqi_pakistan.json')

if not RAW_WEATHER.exists() or not RAW_AQI.exists():
    print("Raw JSON files not found — skipping duplicate check.")
    print("Run the ingestion script first to generate raw/weather_pakistan.json")
else:
    with open(RAW_WEATHER) as f:
        raw_weather = json.load(f)
    with open(RAW_AQI) as f:
        raw_aqi = json.load(f)

    # Build weather df
    weather_dfs = []
    for city_name, payload in raw_weather.items():
        meta   = payload['meta']
        hourly = payload['data']['hourly']
        df = (
            pl.DataFrame(hourly)
            .with_columns([
                pl.col('time').str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M')
                  .dt.replace_time_zone('Asia/Karachi'),
                pl.lit(meta['geonameid']).alias('geonameid'),
            ])
        )
        weather_dfs.append(df)

    weather_df = pl.concat(weather_dfs)

    # Build aqi df
    aqi_dfs = []
    for city_name, payload in raw_aqi.items():
        meta   = payload['meta']
        hourly = payload['data']['hourly']
        df = (
            pl.DataFrame(hourly)
            .with_columns([
                pl.col('time').str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M')
                  .dt.replace_time_zone('Asia/Karachi'),
                pl.lit(meta['geonameid']).alias('geonameid'),
            ])
        )
        aqi_dfs.append(df)

    aqi_df = pl.concat(aqi_dfs)

    print(f"Weather rows: {weather_df.shape[0]:,}")
    print(f"AQI rows:     {aqi_df.shape[0]:,}")
    print()

    # Check duplicates on (time, geonameid)
    weather_dupes = (
        weather_df
        .group_by(['time', 'geonameid'])
        .len()
        .filter(pl.col('len') > 1)
    )
    aqi_dupes = (
        aqi_df
        .group_by(['time', 'geonameid'])
        .len()
        .filter(pl.col('len') > 1)
    )

    print(f"Duplicate (time, geonameid) in weather: {weather_dupes.shape[0]:,}")
    print(f"Duplicate (time, geonameid) in AQI:     {aqi_dupes.shape[0]:,}")

    if weather_dupes.shape[0] > 0:
        print("\nSample weather duplicates:")
        print(weather_dupes.head(5))

    if aqi_dupes.shape[0] > 0:
        print("\nSample AQI duplicates:")
        print(aqi_dupes.head(5))

    # ── 3. Check unique geonameids vs unique city names ───────────────────────
    print()
    print("=" * 60)
    print("3. GEONAMEID vs CITY NAME UNIQUENESS")
    print("=" * 60)

    unique_geonameids  = weather_df.select('geonameid').unique().shape[0]
    unique_city_names  = len(set(raw_weather.keys()))
    print(f"Unique geonameids:  {unique_geonameids}")
    print(f"Unique city names:  {unique_city_names}")

    if unique_geonameids != unique_city_names:
        print("⚠ Mismatch — multiple cities sharing a geonameid or vice versa")

        # Find which city names map to same geonameid
        gid_to_names = defaultdict(list)
        for city_name, payload in raw_weather.items():
            gid_to_names[payload['meta']['geonameid']].append(city_name)

        duped_gids = {gid: names for gid, names in gid_to_names.items() if len(names) > 1}
        if duped_gids:
            print("\nGeonameids with multiple city names:")
            for gid, names in list(duped_gids.items())[:10]:
                print(f"  {gid} → {names}")
    else:
        print("✓ Each geonameid maps to exactly one city name")

print()
print("=" * 60)
print("Done.")
print("=" * 60)