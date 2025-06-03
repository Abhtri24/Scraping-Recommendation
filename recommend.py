import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    # Clean price data
    df['Price'] = df['Price'].apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))) if x != 'N/A' else np.nan)
    
    # Clean rating data
    df['Rating'] = df['Rating'].apply(lambda x: float(x.split()[0]) if isinstance(x, str) and x != 'N/A' else np.nan)
    
    # Create a combined text feature for each laptop
    df['Features'] = df['Name'] + ' ' + df['Brand']
    
    # Add specifications to features
    spec_columns = ['Processor', 'RAM', 'Storage', 'Display', 'Graphics', 'Operating_System']
    for col in spec_columns:
        df['Features'] += ' ' + df[col].fillna('')
    
    return df

def create_recommendation_system():
    # Load the data
    try:
        df = pd.read_csv('amazon_laptops.csv')
        print(f"Loaded {len(df)} laptops from the dataset")
    except FileNotFoundError:
        print("Error: amazon_laptops.csv not found. Please run the scraper first.")
        return None
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Features'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

def get_recommendations(laptop_name, df, cosine_sim, n_recommendations=5, price_range=None, min_rating=None):
    # Find the index of the laptop
    try:
        idx = df[df['Name'].str.contains(laptop_name, case=False, na=False)].index[0]
    except IndexError:
        print(f"Laptop '{laptop_name}' not found in the dataset.")
        return None
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get laptop indices
    laptop_indices = [i[0] for i in sim_scores[1:]]  # Skip the first one (itself)
    
    # Get recommendations
    recommendations = df.iloc[laptop_indices][['Name', 'Price', 'Rating', 'Brand', 
                                             'Processor', 'RAM', 'Storage', 'Display', 
                                             'Graphics', 'Operating_System']]
    
    # Apply filters if specified
    if price_range:
        min_price, max_price = price_range
        recommendations = recommendations[
            (recommendations['Price'] >= min_price) & 
            (recommendations['Price'] <= max_price)
        ]
    
    if min_rating:
        recommendations = recommendations[recommendations['Rating'] >= min_rating]
    
    # Return top N recommendations
    return recommendations.head(n_recommendations)

def analyze_dataset(df):
    print("\nDataset Analysis:")
    print(f"Total laptops: {len(df)}")
    print(f"Unique brands: {df['Brand'].nunique()}")
    print("\nPrice Statistics:")
    print(df['Price'].describe())
    print("\nRating Statistics:")
    print(df['Rating'].describe())
    print("\nTop 5 Brands by Count:")
    print(df['Brand'].value_counts().head())
    print("\nProcessor Types Distribution:")
    print(df['Processor'].value_counts().head())
    print("\nRAM Distribution:")
    print(df['RAM'].value_counts().head())

def main():
    # Create recommendation system
    result = create_recommendation_system()
    if result is None:
        return
    
    df, cosine_sim = result
    
    while True:
        print("\nLaptop Recommendation System")
        print("1. Get recommendations for a laptop")
        print("2. Show all laptops in dataset")
        print("3. Analyze dataset")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            laptop_name = input("\nEnter laptop name (or part of the name): ")
            
            # Get price range
            price_range = None
            use_price_filter = input("Do you want to filter by price range? (y/n): ").lower()
            if use_price_filter == 'y':
                min_price = float(input("Enter minimum price: "))
                max_price = float(input("Enter maximum price: "))
                price_range = (min_price, max_price)
            
            # Get minimum rating
            min_rating = None
            use_rating_filter = input("Do you want to filter by minimum rating? (y/n): ").lower()
            if use_rating_filter == 'y':
                min_rating = float(input("Enter minimum rating (1-5): "))
            
            recommendations = get_recommendations(laptop_name, df, cosine_sim, 
                                               price_range=price_range, 
                                               min_rating=min_rating)
            if recommendations is not None:
                print("\nRecommended laptops:")
                print(recommendations.to_string(index=False))
        
        elif choice == '2':
            print("\nAll laptops in dataset:")
            print(df[['Name', 'Price', 'Rating', 'Brand', 
                     'Processor', 'RAM', 'Storage']].to_string(index=False))
        
        elif choice == '3':
            analyze_dataset(df)
        
        elif choice == '4':
            print("\nThank you for using the recommendation system!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main() 