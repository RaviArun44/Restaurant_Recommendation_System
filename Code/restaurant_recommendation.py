import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from fuzzywuzzy import process
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Initialize geolocator
geolocator = Nominatim(user_agent="restaurant-recommender")

def get_country(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        if location and 'country' in location.raw['address']:
            return location.raw['address']['country']
        else:
            return None
    except GeocoderTimedOut:
        return None

# Paths to Yelp dataset files
business_path = r"C:\Users\MUIS\OneDrive - Eltronic Group A S\Desktop\ArunProject\yelp_academic_dataset_business.json"
review_path = r"C:\Users\MUIS\OneDrive - Eltronic Group A S\Desktop\ArunProject\yelp_academic_dataset_review.json"

# Step 1: Load restaurant businesses
#print("Loading business data...")
restaurants = []
with open(business_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get("categories") and "Restaurants" in data["categories"]:
            restaurants.append(data)
restaurants_df = pd.DataFrame(restaurants)
restaurant_ids = set(restaurants_df['business_id'])
#print(f"Loaded {len(restaurants_df)} restaurant businesses.")

# Step 2: Load reviews for restaurant businesses
#print("Loading reviews...")
review_chunk = []
with open(review_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        #if i % 100000 == 0 and i > 0:
            #print(f"Processed {i} reviews...")
        data = json.loads(line)
        if data['business_id'] in restaurant_ids:
            review_chunk.append(data)
        if len(review_chunk) >= 100000:
            break
reviews_df = pd.DataFrame(review_chunk)
#print(f"Loaded {len(reviews_df)} reviews related to restaurants.")

# Convert review date to datetime and clean up
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_df = reviews_df[['user_id', 'business_id', 'stars', 'text', 'date']]

# Merge with business info
merged_df = pd.merge(
    reviews_df,
    restaurants_df[['business_id', 'name', 'categories', 'city', 'state', 'latitude', 'longitude']],
    on='business_id',
    how='left'
)
print(f"Merged dataset shape: {merged_df.shape}")

# Create metadata
restaurant_metadata = merged_df[['name', 'city', 'state', 'latitude', 'longitude']].drop_duplicates().set_index('name')
restaurant_metadata.to_pickle("restaurant_metadata.pkl")
print("Restaurant_metadata saved as 'restaurant_metadata.pkl'")

# Create user-item rating matrix
user_item_matrix = merged_df.pivot_table(index='user_id', columns='name', values='stars')
user_item_matrix_filled = user_item_matrix.fillna(0)

# Apply Truncated SVD
svd = TruncatedSVD(n_components=20, random_state=42)
item_matrix = svd.fit_transform(user_item_matrix_filled.T)

# Compute cosine similarity between restaurants
similarity_matrix = cosine_similarity(item_matrix)
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_item_matrix_filled.columns,
    columns=user_item_matrix_filled.columns
)
similarity_df.to_pickle("similarity_matrix.pkl")
print("Similarity matrix saved as 'similarity_matrix.pkl'")

# --- Recommendation Logic ---

def get_best_match(query, choices, threshold=70):
    match, score = process.extractOne(query, choices)
    return match if score >= threshold else None

def recommend_similar_restaurants_svd(target_restaurant, similarity_df, metadata_df, n=5, save_csv=False):
    best_match = get_best_match(target_restaurant, similarity_df.index)
    if not best_match:
        print(f"Restaurant '{target_restaurant}' not found (even after fuzzy match).")
        return pd.DataFrame()

    similar_scores = similarity_df[best_match].drop(best_match)
    top_similar = similar_scores.sort_values(ascending=False).head(n)
    similarity_percent = (top_similar.values * 100).round(1)

    meta_info = metadata_df.loc[top_similar.index]

    # Get country for each result
    countries = [
        get_country(lat, lon)
        for lat, lon in zip(meta_info['latitude'], meta_info['longitude'])
    ]

    result_df = pd.DataFrame({
        "Similar Restaurant": top_similar.index,
        "City": meta_info['city'].values,
        "State": meta_info['state'].values,
        "Country": countries,
        "Similarity Score (%)": similarity_percent
    })

    print(f"\nTop {n} restaurants similar to '{best_match}':")
    print(result_df)

    if save_csv:
        filename = f"top_{n}_similar_to_{best_match.replace(' ', '_')}.csv"
        result_df.to_csv(filename, index=False)
        print(f"Saved to '{filename}'")

    return result_df

# Example usage:
if __name__ == "__main__":
    recommend_similar_restaurants_svd("1 Stop Piza", similarity_df, restaurant_metadata, n=5, save_csv=True)
