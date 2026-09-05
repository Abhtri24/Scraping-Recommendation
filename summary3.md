# Project Overview

The repository implements a laptop recommendation system built with Python and Streamlit. Data is collected from Amazon using `scrape.py`, which leverages Selenium and BeautifulSoup to navigate product pages, extract specifications such as processor, RAM, storage, display, graphics, operating system, price, and rating, and store the results in `amazon_laptops.csv`. The CSV serves as the primary dataset for the recommendation engine.

`recommend.py` contains the core logic for transforming the raw data into a searchable feature space. Textual attributes (e.g., processor, brand, graphics) are vectorized with `TfidfVectorizer`, while numeric fields (price, rating, RAM, storage) are cleaned, converted to appropriate types, and scaled using `MinMaxScaler`. The module defines a `preprocess_data` function that normalizes the dataset and a `get_recommendations` function that computes cosine similarity between a user‑selected laptop and all others, returning the most similar items.

The front‑end is defined in `app.py`. Streamlit widgets allow users to upload a CSV, select a laptop model, and adjust the number of recommendations. Visualizations generated with Matplotlib and Seaborn display price distributions, rating histograms, and feature correlations. The app also shows a table of recommended laptops with key specifications and a downloadable CSV of the results.

The `requirements.txt` lists all third‑party libraries needed to run the project, including `pandas`, `numpy`, `scikit-learn`, `beautifulsoup4`, `selenium`, and `streamlit`. The repository is structured for easy extension: new data sources can be added to `scrape.py`, additional similarity metrics can be incorporated in `recommend.py`, and the UI can be expanded in `app.py`.

Overall, the project demonstrates a complete pipeline—from web scraping and data cleaning to machine‑learning‑based similarity scoring and interactive visualization—providing users with personalized laptop recommendations based on both technical specifications and price considerations.
