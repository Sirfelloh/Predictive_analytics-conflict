# scrape_and_predict.py

import requests
from bs4 import BeautifulSoup
import re
import pickle

# Load the model and vectorizer
with open('civil_unrest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def scrape_article_text(url):
    """
    Attempts to scrape the main article text from a web page by checking
    common container classes, <article> tags, or falling back to <p> tags.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Attempt known selectors
        for selector in ['article-body', 'main-content', 'post-content', 'article-content']:
            main_content_div = soup.find('div', class_=selector)
            if main_content_div:
                return main_content_div.get_text(separator=' ', strip=True)

        # Attempt <article> tag if present
        article_tag = soup.find('article')
        if article_tag:
            return article_tag.get_text(separator=' ', strip=True)

        # Fallback: join all <p> tags
        paragraphs = soup.find_all('p')
        text = ' '.join(para.get_text() for para in paragraphs)
        return text

    except requests.exceptions.RequestException:
        return ""  # Return empty string if request fails

def preprocess_text(text):
    """
    Basic text preprocessing: lowercasing, removing URLs and non-alphanumeric chars, etc.
    """
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_unrest(url):
    """
    Main convenience function that:
    1. Scrapes article text
    2. Preprocesses it
    3. Vectorizes
    4. Predicts using the loaded model
    Returns (raw_text, cleaned_text, prediction_label)
    """
    raw_text = scrape_article_text(url)
    cleaned = preprocess_text(raw_text)
    if not cleaned.strip():
        return raw_text, cleaned, None  # or some placeholder

    vectorized_data = vectorizer.transform([cleaned])
    pred = model.predict(vectorized_data)[0]

    # Convert numeric label to string label as needed
    label = "Civil Unrest" if pred == 1 else "No Civil Unrest"
    return raw_text, cleaned, label
