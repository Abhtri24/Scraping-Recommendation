import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Laptop Recommendation System", page_icon="💻", layout="wide")

st.title("💻 Laptop Recommendation System")
st.markdown("""
This app recommends laptops based on your preferences. Use the sidebar to filter and explore suggestions.
""")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("amazon_laptops.csv")

        df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df['Rating'] = pd.to_numeric(df['Rating'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0], errors='coerce')
        df['Rating'] = df['Rating'].fillna(3.5).clip(0, 5)
        df = df.dropna(subset=['Price'])

        for col in ['Brand', 'Processor', 'RAM', 'Storage', 'Display', 'Graphics', 'Operating_System']:
            df[col] = df[col].fillna('Unknown')

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def build_knn_model(df):
    text_features = df[['Brand', 'Processor', 'RAM', 'Storage', 'Display', 'Graphics', 'Operating_System']].agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text_features)

    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(df[['Price', 'Rating']])

    from scipy.sparse import hstack
    final_features = hstack([tfidf_matrix, numeric_features])
    
    # Convert sparse matrix to dense array
    final_features_dense = final_features.toarray()

    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(final_features_dense)

    return knn, final_features_dense

def get_knn_recommendations(df, knn_model, features, laptop_name):
    idx = df[df['Name'] == laptop_name].index[0]
    # Use the dense array directly
    distances, indices = knn_model.kneighbors(features[idx].reshape(1, -1), n_neighbors=6)
    recommended_indices = [i for i in indices[0] if i != idx][:5]
    return df.iloc[recommended_indices]

df = load_data()

if not df.empty:
    knn_model, features = build_knn_model(df)

    st.sidebar.header("Filters")
    min_price, max_price = int(df['Price'].min()), int(df['Price'].max())
    price_range = st.sidebar.slider("Price Range (₹)", min_price, max_price, (min_price, max_price))
    brand = st.sidebar.selectbox("Brand", ['All'] + sorted(df['Brand'].unique()))
    ram = st.sidebar.selectbox("RAM", ['All'] + sorted(df['RAM'].unique()))
    storage = st.sidebar.selectbox("Storage", ['All'] + sorted(df['Storage'].unique()))
    min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)

    filtered_df = df[
        (df['Price'] >= price_range[0]) &
        (df['Price'] <= price_range[1]) &
        (df['Rating'] >= min_rating)
    ]

    if brand != 'All':
        filtered_df = filtered_df[filtered_df['Brand'] == brand]
    if ram != 'All':
        filtered_df = filtered_df[filtered_df['RAM'] == ram]
    if storage != 'All':
        filtered_df = filtered_df[filtered_df['Storage'] == storage]

    if not filtered_df.empty:
        st.metric("Total Laptops", len(filtered_df))
        st.metric("Avg Price", f"₹{filtered_df['Price'].mean():,.2f}")
        st.metric("Avg Rating", f"{filtered_df['Rating'].mean():.2f}")

        st.subheader("🔍 Recommendations")
        selected_laptop = st.selectbox("Choose a laptop to find similar ones:", filtered_df['Name'].tolist())
        if selected_laptop:
            recommendations = get_knn_recommendations(df, knn_model, features, selected_laptop)
            for _, row in recommendations.iterrows():
                with st.expander(f"{row['Name']} - ₹{row['Price']:,.2f}"):
                    st.write(f"**Brand:** {row['Brand']} | **Processor:** {row['Processor']} | **RAM:** {row['RAM']}")
                    st.write(f"**Storage:** {row['Storage']} | **Graphics:** {row['Graphics']} | **OS:** {row['Operating_System']}")
                    st.write(f"**Rating:** {row['Rating']} ⭐")

        st.subheader("📊 Data Insights")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['Price'], bins=30, ax=ax)
        plt.title('Price Distribution')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.scatterplot(x='Price', y='Rating', data=filtered_df, ax=ax)
        plt.title('Price vs Rating')
        st.pyplot(fig)

        st.subheader("Laptop Listings")
        st.dataframe(filtered_df.reset_index(drop=True))
    else:
        st.warning("No laptops found with current filters.")
else:
    st.error("No data available to display.")
