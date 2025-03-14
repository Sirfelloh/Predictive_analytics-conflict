# test_standalone.py

from url_crawler import *

article_url = "https://www.tuko.co.ke/kenya/576828-thika-road-protesting-kenyatta-university-students-block-section-highway-power-outage/"
article_text = scrape_article_text(article_url)
getPredictionResult(preprocess_text, model, vectorizer, article_text)
