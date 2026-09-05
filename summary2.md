# Project Overview

This repository implements a laptop recommendation system that scrapes laptop data from Amazon, processes the information, and provides personalized suggestions through an interactive Streamlit web application. The core components are organized into distinct Python scripts:

- **scrape.py**: Utilizes Selenium and BeautifulSoup to navigate Amazon product pages, extract specifications such as processor, RAM, storage, display, graphics, and operating system, and store the results in `amazon_laptops.csv`. It handles dynamic content loading, pagination, and basic data cleaning.

- **recommend.py**: Contains the recommendation logic. It loads the CSV, preprocesses textual and numeric features, and creates a TF‑IDF matrix for textual attributes (e.g., name, brand). Numeric fields like price and rating are scaled with `MinMaxScaler`. Cosine similarity between the target laptop and all others is computed, and the nearest neighbors are returned as recommendations.

- **app.py**: Provides the front‑end using Streamlit. Users can upload a CSV, select a laptop, adjust weighting for price, rating, and specifications, and view recommended alternatives. Visualizations (histograms, scatter plots) are generated with Matplotlib and Seaborn to help users explore the dataset.

- **requirements.txt**: Lists all dependencies, including `pandas`, `numpy`, `scikit-learn`, `beautifulsoup4`, `selenium`, and `streamlit`.

The workflow follows these steps:
1. **Data Collection** – Run `scrape.py` to generate up‑to‑date laptop data.
2. **Data Preparation** – `recommend.py` cleans and encodes the data, handling missing values and normalizing numeric columns.
3. **Recommendation Engine** – Similarity scores are calculated, and the top N similar laptops are presented.
4. **User Interaction** – `app.py` offers an intuitive UI where users can filter results, view detailed specs, and compare price‑performance trade‑offs.

Overall, the project demonstrates end‑to‑end data engineering, machine‑learning based similarity matching, and a user‑friendly web interface, making it a practical tool for consumers seeking laptop alternatives based on their preferences.
