import os
import streamlit as st
import pandas as pd
from fuzzywuzzy import process

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
similarity_df = pd.read_pickle(os.path.join(BASE_DIR, "similarity_matrix.pkl"))
#restaurant_metadata = pd.read_pickle(os.path.join(BASE_DIR, "restaurant_metadata.pkl"))
restaurant_metadata = pd.read_pickle(os.path.join(BASE_DIR, "restaurant_metadata_with_country.pkl"))

# Ensure all required columns are in metadata
required_columns = {"city", "state", "country", "latitude", "longitude"}
missing = required_columns - set(restaurant_metadata.columns)
if missing:
    st.error(f"Metadata is missing required columns: {', '.join(missing)}")
    st.stop()

# Functions
def get_best_match(query, choices, threshold=70):
    match, score = process.extractOne(query, choices)
    return match if score >= threshold else None

def recommend_similar_restaurants_svd(target_restaurant, similarity_df, metadata_df, n=5, city_filter=None, country_filter=None):
    best_match = get_best_match(target_restaurant, similarity_df.index)
    if not best_match:
        return pd.DataFrame()

    similar_scores = similarity_df[best_match].drop(best_match)
    top_similar = similar_scores.sort_values(ascending=False)

    meta_info = metadata_df.loc[top_similar.index]

    # Apply filters
    if country_filter:
        meta_info = meta_info[meta_info['country'] == country_filter]
    if city_filter:
        meta_info = meta_info[meta_info['city'] == city_filter]

    # Final top N after filtering
    filtered_index = meta_info.index
    top_similar = top_similar.loc[filtered_index].head(n)
    meta_info = meta_info.loc[top_similar.index]
    similarity_percent = (top_similar.values * 100).round(1)

    result_df = pd.DataFrame({
        "🍴 Similar Restaurant": top_similar.index,
        "📍 City": meta_info['city'].values,
        "🏙️ State": meta_info['state'].values,
        "🌍 Country": meta_info['country'].values,
        "🌍 Country": meta_info['country'].values,
        "💯 Similarity Score (%)": similarity_percent
    })

    # Optional: include cuisine if available
    if 'categories' in meta_info.columns:
        result_df["🍱 Cuisine"] = meta_info['categories'].values

    return result_df

# Page setup
st.set_page_config(page_title="Restaurant Recommender 🍽️", page_icon="🍕", layout="wide")

# Custom styles and background
st.markdown("""
    <style>
        .stApp {
            background-color: #fff8f0;
            font-family: 'Segoe UI', sans-serif;
        }
        .big-font {
            font-size: 36px !important;
            font-weight: bold;
            color: #d62828;
            text-align: center;
            margin-bottom: 20px;
        }
        .section {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Centered Caption Without Image
st.markdown("""
    <div style='text-align: center; margin-bottom: 10px;'>
        <h3 style='font-size: 24px; color: #d62828;'>🍕 Find Your Next Favorite Spot!</h3>
    </div>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-font">Restaurant Recommendation System</div>', unsafe_allow_html=True)

# Title
#st.markdown('<div class="big-font">🍔 Restaurant Recommendation System</div>', unsafe_allow_html=True)
st.markdown("🔎 Discover restaurants similar to your favorite ones. Powered by collaborative filtering and fuzzy matching!")

# User Input
with st.container():
    st.markdown("### Input your Favourite Restaurant")
    col1, col2 = st.columns([3, 1])
    with col1:
        restaurant_input = st.text_input("🍜 Restaurant name", placeholder="e.g. Pizza Palace")
    with col2:
        top_n = st.slider("Number of Recommendations", 1, 10, 5)

# Location filters
st.markdown("### 🧭 Refine by Location")
col1, col2 = st.columns(2)

with col1:
    selected_country = st.selectbox("Country", options=["All"] + sorted(restaurant_metadata['country'].dropna().unique().tolist()))
    if selected_country != "All":
        country_filtered_df = restaurant_metadata[restaurant_metadata["country"] == selected_country]
    else:
        country_filtered_df = restaurant_metadata

with col2:
    selected_city = st.selectbox("City", options=["All"] + sorted(country_filtered_df['city'].dropna().unique().tolist()))

# Suggestion Help
with st.expander("💡 Need Suggestions?"):
    st.markdown("Try one of these restaurant names:")
    sample_names = list(similarity_df.index[:10])
    st.code(", ".join(sample_names), language='plaintext')

# Button and recommendation logic
if st.button("🍽️ Recommend Similar Restaurants"):
    if restaurant_input:
        with st.spinner("🍳 Cooking up your results..."):
            city_filter = selected_city if selected_city != "All" else None
            country_filter = selected_country if selected_country != "All" else None

            result = recommend_similar_restaurants_svd(
                restaurant_input, similarity_df, restaurant_metadata,
                n=top_n, city_filter=city_filter, country_filter=country_filter
            )

        if result.empty:
            st.error("⚠️ No similar restaurants found with current filters. Try a different name or relax filters.")
        else:
            st.success(f"🥂 Top {len(result)} similar restaurants to **{restaurant_input.title()}**")
            st.dataframe(result, use_container_width=True)

            # Download button
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Recommendations as CSV", csv, f"recommendations_{restaurant_input}.csv", "text/csv")

            # Map
            if {"latitude", "longitude"}.issubset(restaurant_metadata.columns):
                try:
                    map_data = restaurant_metadata.loc[result["🍴 Similar Restaurant"]][["latitude", "longitude"]]
                    st.map(map_data)
                except Exception as e:
                    st.warning(f"Map display failed: {e}")
    else:
        st.info("Please enter a restaurant name to begin.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for food lovers")
