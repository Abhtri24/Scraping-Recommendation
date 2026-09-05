# Laptop Recommendation System

This repository implements a complete end‑to‑end laptop recommendation web application built with **Streamlit**. The solution is organized into three primary components: data acquisition, data preprocessing & modeling, and an interactive user interface.

## 1. Data Collection (`scrape.py`)
The `scrape.py` script leverages **BeautifulSoup** for HTML parsing and **Selenium** (via `webdriver_manager`) to handle dynamic content on Amazon product pages. It navigates through laptop listings, extracts key attributes such as product title, price, processor, RAM, storage, graphics, screen size, and rating, and writes the structured data to `amazon_laptops.csv`. The script includes basic error handling, rate‑limiting via `time.sleep`, and timestamped logging to ensure reproducibility.

## 2. Data Preparation & Recommendation Logic (`recommend.py`)
`recommend.py` reads the CSV, cleans monetary values, normalizes numeric fields, and tokenizes textual specifications. It employs a **TF‑IDF** vectorizer to capture textual similarity across product descriptions and a **MinMaxScaler** to bring numeric features onto a comparable scale. Cosine similarity is computed between the target laptop and the rest of the catalog, and the top‑N most similar items are returned. The module also provides helper functions for feature engineering, such as extracting numeric values from strings and handling missing data.

## 3. Interactive Front‑End (`app.py`)
The Streamlit app (`app.py`) presents a sleek, responsive UI where users can either select a laptop from a dropdown or upload their own specifications. Upon submission, the app calls the recommendation engine, displays a ranked list of similar laptops, and visualizes key comparisons using **Matplotlib** and **Seaborn** (e.g., price distribution, performance metrics). The layout is configured for wide screens, includes custom page icons, and uses caching to speed up repeated queries.

## Project Structure
- `amazon_laptops.csv` – Raw scraped dataset.
- `scrape.py` – Scraper script.
- `recommend.py` – Data preprocessing and similarity calculations.
- `app.py` – Streamlit UI.
- `requirements.txt` – Python dependencies.
- `summary.md` – This documentation file.

Overall, the project demonstrates how to combine web scraping, machine‑learning preprocessing, and a lightweight web framework to deliver personalized product recommendations in a user‑friendly manner.