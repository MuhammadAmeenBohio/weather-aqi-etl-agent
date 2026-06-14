import geonamescache
 
 
def get_pk_cities() -> list[dict]:
    """
    Returns a list of dicts, one per Pakistan city:
      geonameid, city, province, latitude, longitude, population, timezone
    """
    gc = geonamescache.GeonamesCache()
    all_cities = gc.get_cities()
 
    return [
        {
            'geonameid': gid,
            'city':      city['name'],
            'province':  city['admin1code'],
            'latitude':  float(city['latitude']),
            'longitude': float(city['longitude']),
            'population': city['population'],
            'timezone':  city['timezone'],
        }
        for gid, city in all_cities.items()
        if city['countrycode'] == 'PK'
    ]