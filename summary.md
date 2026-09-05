# Laptop Recommendation System

This repository implements a laptop recommendation web application built with **Streamlit**. The workflow consists of three main components:

1. **Data collection (`scrape.py`)** – Uses **BeautifulSoup** and **Selenium** to crawl Amazon laptop listings, extracts specifications (price, processor, RAM, storage, etc.) and stores them in `amazon_laptops.csv`.

2. **Recommendation engine (`recommend.py`)** – Loads the CSV, cleans and normalizes the data, creates a TF‑IDF representation of textual specs, scales numeric features, and computes similarity scores with **cosine similarity** and **NearestNeighbors**. The `preprocess_data` function prepares the dataset, while `get_recommendations` returns the most similar laptops for a given query.

3. **User interface (`app.py`)** – A Streamlit front‑end that lets users select desired criteria (price range, brand, processor, etc.). The UI calls the functions from `recommend.py` and displays the top recommendations together with visualisations (price distribution, feature comparisons) using **matplotlib** and **seaborn**.

The `requirements.txt` lists all Python dependencies required to run the scraper, the recommendation engine, and the Streamlit app.

Overall, the project demonstrates a complete pipeline from web scraping to a machine‑learning‑driven recommendation system that can be launched locally with `streamlit run app.py`. 