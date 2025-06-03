import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import random
import os
import re

def extract_specs(text):
    specs = {
        'Processor': None,
        'RAM': None,
        'Storage': None,
        'Display': None,
        'Graphics': None,
        'Operating_System': None
    }
    
    # Extract processor
    processor_patterns = [
        r'(Intel|AMD)\s+(Core|Ryzen|Pentium|Celeron)\s+[A-Za-z0-9\s-]+',
        r'(Intel|AMD)\s+[A-Za-z0-9\s-]+(Processor|CPU)'
    ]
    for pattern in processor_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['Processor'] = match.group(0)
            break
    
    # Extract RAM
    ram_patterns = [
        r'(\d+)\s*GB\s*(?:DDR\d*)?\s*RAM',
        r'(\d+)\s*GB\s*Memory'
    ]
    for pattern in ram_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['RAM'] = f"{match.group(1)} GB"
            break
    
    # Extract Storage
    storage_patterns = [
        r'(\d+)\s*(?:GB|TB)\s*(?:SSD|HDD|Storage)',
        r'(\d+)\s*(?:GB|TB)\s*(?:Solid\s*State|Hard\s*Drive)'
    ]
    for pattern in storage_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['Storage'] = match.group(0)
            break
    
    # Extract Display
    display_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:inch|")\s*(?:Display|Screen)',
        r'(\d+(?:\.\d+)?)\s*(?:inch|")\s*(?:HD|FHD|UHD|4K)'
    ]
    for pattern in display_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['Display'] = match.group(0)
            break
    
    # Extract Graphics
    graphics_patterns = [
        r'(NVIDIA|AMD|Intel)\s+[A-Za-z0-9\s-]+(?:Graphics|GPU)',
        r'(NVIDIA|AMD|Intel)\s+[A-Za-z0-9\s-]+(?:MX|GTX|RTX)'
    ]
    for pattern in graphics_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['Graphics'] = match.group(0)
            break
    
    # Extract OS
    os_patterns = [
        r'(Windows\s+\d+\s*(?:Home|Pro)?)',
        r'(Windows\s+\d+)',
        r'(Chrome\s*OS)',
        r'(Linux)'
    ]
    for pattern in os_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['Operating_System'] = match.group(0)
            break
    
    return specs

def scrape_amazon_laptops(max_pages=10):
    # Remove existing CSV file if it exists
    if os.path.exists('amazon_laptops.csv'):
        os.remove('amazon_laptops.csv')
        print("Removed existing amazon_laptops.csv file")
    
    # Initialize empty list for data
    laptops_data = []
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    # Add more realistic browser behavior
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)
    
    # Set user agent
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    try:
        current_page = 1
        
        while current_page <= max_pages:
            # URL for Amazon laptops with page number
            url = f"https://www.amazon.in/s?k=laptops&page={current_page}"
            print(f"\nNavigating to page {current_page}...")
            driver.get(url)
            
            # Wait for the page to load
            print("Waiting for page to load...")
            time.sleep(random.uniform(3, 5))
            
            # Wait for search results to be visible
            wait = WebDriverWait(driver, 10)
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-component-type="s-search-result"]')))
            except TimeoutException:
                print("Timeout waiting for search results. Trying alternative selector...")
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.s-result-item')))
                except TimeoutException:
                    print("Could not find any products. The page might be blocked.")
                    break
            
            # Find all laptop items on the page
            items = driver.find_elements(By.CSS_SELECTOR, 'div[data-component-type="s-search-result"]')
            if not items:
                items = driver.find_elements(By.CSS_SELECTOR, '.s-result-item')
            
            print(f"Found {len(items)} items on page {current_page}")
            
            if not items:
                print("No items found on this page. Stopping pagination.")
                break
            
            page_laptops = 0
            for item in items:
                try:
                    # Extract name
                    try:
                        name_element = item.find_element(By.CSS_SELECTOR, 'h2 .a-text-normal')
                    except NoSuchElementException:
                        try:
                            name_element = item.find_element(By.CSS_SELECTOR, '.a-size-medium')
                        except NoSuchElementException:
                            continue
                    
                    name = name_element.text.strip()
                    if not name:
                        continue
                    
                    # Extract price
                    try:
                        price_element = item.find_element(By.CSS_SELECTOR, '.a-price-whole')
                        price = price_element.text.strip()
                    except NoSuchElementException:
                        try:
                            price_element = item.find_element(By.CSS_SELECTOR, '.a-offscreen')
                            price = price_element.text.strip()
                        except NoSuchElementException:
                            price = 'N/A'
                    
                    # Extract rating
                    try:
                        rating_element = item.find_element(By.CSS_SELECTOR, '.a-icon-star-small')
                        rating = rating_element.text.strip()
                    except NoSuchElementException:
                        try:
                            rating_element = item.find_element(By.CSS_SELECTOR, '.a-icon-star')
                            rating = rating_element.text.strip()
                        except NoSuchElementException:
                            rating = 'N/A'
                    
                    # Extract brand
                    try:
                        brand_element = item.find_element(By.CSS_SELECTOR, '.a-size-base-plus')
                        brand = brand_element.text.strip()
                    except NoSuchElementException:
                        brand = name.split()[0] if name else 'N/A'
                    
                    # Extract specifications
                    specs = extract_specs(name)
                    
                    # Add laptop to data
                    laptop_data = {
                        'Name': name,
                        'Price': price,
                        'Rating': rating,
                        'Brand': brand,
                        'Page': current_page,
                        **specs  # Add all specifications
                    }
                    laptops_data.append(laptop_data)
                    page_laptops += 1
                    print(f"Added laptop: {name[:50]}...")
                    
                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    continue
            
            print(f"Added {page_laptops} laptops from page {current_page}")
            
            # Save progress after each page
            if laptops_data:
                df = pd.DataFrame(laptops_data)
                df.to_csv('amazon_laptops.csv', index=False)
                print(f"\nProgress saved: {len(laptops_data)} laptops scraped so far")
            
            # Random delay before next page
            time.sleep(random.uniform(2, 4))
            current_page += 1
        
        if laptops_data:
            print(f"\nScraping completed! Total laptops scraped: {len(laptops_data)}")
            print(f"Data saved to amazon_laptops.csv")
        else:
            print("\nNo data was scraped. The website structure might have changed or the request was blocked.")
        
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    
    finally:
        # Always close the browser
        driver.quit()

if __name__ == "__main__":
    scrape_amazon_laptops(max_pages=10)
