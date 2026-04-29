import hashlib
import os
import requests
import random
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"

def get_paged_url(search_query, page_num=0):
    """
    Example
    URL:
    https://jp.mercari.com/search?keyword=%E3%81%93%E3%81%A8%E3%81%AD%E3%80%80%E3%81%AC%E3%81%84&page_token=v1%3A2
    
    page num is 0-based index
    """
    
    if page_num < 0:
        raise ValueError("Page number must not be negative.")
    
    base_url = f"https://jp.mercari.com/search?keyword={search_query}"
    if page_num == 0:
        return base_url
    else:
        return f"{base_url}&page_token=v1:{page_num}"

def scrape_nui_mercari(search_query, max_images=100):
    """Scrapes image URLs from Mercari search results for the given query and saves the images.
    Sleep around 3 seconds with random jitter between requests.
    
    Caveat: max_images doesn't consider if new images are duplicated with the existing ones.
    So the actual number of saved images may be less than max_images. Anyway the number of images stored in data directory is around max_images.
    """
    with sync_playwright() as p:
        # headless is recommended for WSL
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=UA)
        page = context.new_page()
        
        image_urls = set()
        
        # Arbitrary limit 10: 10 pages should be enough to get 100 images.
        for page_num in range(10):
            # Open a page
            search_url = get_paged_url(search_query, page_num)
            print(f"Searching for: {search_query} in the {page_num} page...")
            page.goto(search_url)
            time.sleep(3 + random.random() * 2)
            print(page)

            image_elements = page.query_selector_all('img')
            for img in image_elements:
                src = img.get_attribute('src')
                # Exclude ads and icon URLs
                if src and src.startswith('http'):
                    image_urls.add(src)
                    
            if len(image_urls) >= max_images:
                break 

        print(f"Found {len(image_urls)} image URLs!")
        for i, url in enumerate(image_urls):
            print(f"{i+1}: {url[:60]}...")

        browser.close()
        
    save_images(list(image_urls), search_query)

def save_images(image_urls, category_name):
    """Downloads and saves images from the provided URLs.
    Creates a directory named after the category if it doesn't exist, and saves images using a hash of their content to avoid duplicates.
    The hash logic is implemented in the save_image_with_hash function.
    """
    
    save_dir = Path(f"data/raw/{category_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting download for: {category_name}...")

    header = {
        "User-Agent": UA
    }

    for url in image_urls:
        try:
            response = requests.get(url, headers=header, timeout=10)
            response.raise_for_status()

            save_image_with_hash(response.content, category_name)

            # Wait a random time to avoid overwhelming the server
            time.sleep(random.uniform(1.0, 2.0))

        except Exception as e:
            print(f"Failed to download {url}: {e}")
            continue

    print(f"Finished! Total images saved in {category_name}: {len(os.listdir(save_dir))}")

def save_image_with_hash(image_binary, category_name):
    """Saves the image using a hash of its content as the filename to avoid duplicates.
    """
    
    file_hash = hashlib.md5(image_binary).hexdigest()
    
    # Convert all the images to jpg regardless of original format
    # Example: 5d41402abc4b2a76b9719d911017c592.jpg
    file_name = f"{file_hash}.jpg"
    
    save_dir = Path(f"data/raw/{category_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file_name

    if file_path.exists():
        print(f"Skipped (already exists): {file_name}")
        return

    with open(file_path, "wb") as f:
        f.write(image_binary)
    print(f"Saved: {file_name}")

if __name__ == "__main__":
    scrape_nui_mercari("ことね ぬい", max_images=100)
