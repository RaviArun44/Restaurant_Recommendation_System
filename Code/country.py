# country.py

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import os
import time
from tqdm import tqdm

# Caching geocoder
geolocator = Nominatim(user_agent="restaurant_recommender")

def safe_reverse(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        if location and 'country' in location.raw['address']:
            return location.raw['address']['country']
    except GeocoderTimedOut:
        return None
    return None

def enrich_metadata_with_country(metadata_path):
    df = pd.read_pickle(metadata_path)

    # Round coordinates for caching purposes
    df['latlon'] = list(zip(df['latitude'].round(4), df['longitude'].round(4)))

    # Get unique coordinates
    unique_coords = df['latlon'].drop_duplicates()

    # Create latlon → country mapping
    coord_to_country = {}
    print("🌍 Fetching country for each unique lat/lon pair (cached)...")
    for coord in tqdm(unique_coords):
        lat, lon = coord
        coord_to_country[coord] = safe_reverse(lat, lon)
        time.sleep(1)  # Respect rate limits

    df['country'] = df['latlon'].map(coord_to_country)
    df.drop(columns=['latlon'], inplace=True)

    # Save updated metadata
    updated_path = metadata_path.replace(".pkl", "_with_country.pkl")
    df.to_pickle(updated_path)
    print(f"✅ Saved enriched metadata to {updated_path}")

    return updated_path

# Run standalone
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(BASE_DIR, "restaurant_metadata.pkl")
    enrich_metadata_with_country(metadata_path)
