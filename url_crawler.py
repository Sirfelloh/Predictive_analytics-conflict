import requests
from bs4 import BeautifulSoup
import re
import pickle

def scrape_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Common class names or more general selectors for main article content
        for selector in ['article-body', 'main-content', 'post-content', 'article-content']:
            main_content_div = soup.find('div', class_=selector)
            if main_content_div:
                return main_content_div.get_text(separator=' ', strip=True)

        # Try using the <article> tag
        article_tag = soup.find('article')
        if article_tag:
            return article_tag.get_text(separator=' ', strip=True)

        # Fallback to the first <p> if nothing else
        p_tag = soup.find('p')
        if p_tag:
            return p_tag.get_text(separator=' ', strip=True)

        return ""
    except requests.exceptions.RequestException:
        return ""

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the model and vectorizer
with open('civil_unrest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Scrape an article
# article_url = "https://www.tuko.co.ke/people/577358-puzzle-heartbreak-masinde-muliro-university-graduate-dies-month-graduating-sad/"
article_url = "https://www.tuko.co.ke/kenya/576828-thika-road-protesting-kenyatta-university-students-block-section-highway-power-outage/"
article_text = scrape_article_text(article_url)

def getPredictionResult(preprocess_text, model, vectorizer, article_text):
    if article_text:
        print("=== RAW ARTICLE TEXT ===")
        print(article_text[:500], "...\n")  # Print a small preview

    # Preprocess
        cleaned_text = preprocess_text(article_text)
        print("=== CLEANED ARTICLE TEXT ===")
        print(cleaned_text[:500], "...\n")  # Print a small preview

    # Transform for prediction
        article_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(article_vector)[0]
        print("=== MODEL PREDICTION ===")
        print("Label:", prediction)
        return article_text, cleaned_text, prediction
    else:
        print("Failed to scrape article or article text is empty.")

# getPredictionResult(preprocess_text, model, vectorizer, article_text)
