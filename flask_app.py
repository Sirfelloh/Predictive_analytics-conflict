from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import random
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json
from sqlalchemy import func
from itertools import groupby
import pickle
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import feedparser
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Replace with a secure key
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
db = SQLAlchemy(app)

scheduler = BackgroundScheduler()
scheduler.start()

# Load the model and vectorizer with error handling
try:
    with open('civil_unrest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        logger.info("Model loaded successfully!")
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
        logger.info("Vectorizer loaded successfully!")
except FileNotFoundError as e:
    logger.error(f"Failed to load model or vectorizer files: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading model or vectorizer: {e}")
    raise

# Database Models
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location_name = db.Column(db.String(120), nullable=False)
    extra_location = db.Column(db.String(120), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    conflict_level = db.Column(db.Integer, default=0)
    date = db.Column(db.Date, default=datetime.today().date())

    def __repr__(self):
        return f"<Prediction {self.location_name} - {self.extra_location} - Level: {self.conflict_level} on {self.date}>"

class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def __repr__(self):
        return f"<Admin {self.username}>"

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

# Default Admin User Creation
def create_default_admin():
    admin = Admin.query.filter_by(username="admin").first()
    if not admin:
        default_admin = Admin(username="admin")
        default_admin.set_password("adminpassword")
        db.session.add(default_admin)
        db.session.commit()
        logger.info("Default admin created!")

# Real-time Alerts
def check_alerts():
    THRESHOLD = 50
    today = datetime.today().date()
    date_range = [today - timedelta(days=i) for i in range(3)]
    alert_data = []
    for single_date in date_range:
        daily_data = db.session.query(
            Prediction.location_name,
            func.sum(Prediction.conflict_level).label('total_conflict')
        ).filter(Prediction.date == single_date).group_by(Prediction.location_name).all()
        for location_name, total_conflict in daily_data:
            if total_conflict >= THRESHOLD:
                alert_data.append((location_name, single_date, total_conflict))
    if alert_data:
        send_email(alert_data)

def send_email(alert_data):
    sender_email = "your-email@example.com"  # Replace with your email
    receiver_email = "receiver-email@example.com"  # Replace with receiver email
    password = "your-email-password"  # Replace with your email password or app-specific password
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Conflict Alert Summary"
    body = "Alert Summary for the Last 3 Days:\n\n"
    for location, alert_date, conflict_level in alert_data:
        body += f"Location: {location}, Date: {alert_date}, Conflict Level: {conflict_level}\n"
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

scheduler.add_job(
    func=check_alerts,
    trigger=IntervalTrigger(hours=24),
    id='alert_job',
    name='Send daily conflict alerts',
    replace_existing=True
)

# Helper Functions
def scrape_article_text(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for selector in ['article-body', 'main-content', 'post-content', 'article-content']:
            main_content_div = soup.find('div', class_=selector)
            if main_content_div:
                return main_content_div.get_text(separator=' ', strip=True)
        article_tag = soup.find('article')
        if article_tag:
            return article_tag.get_text(separator=' ', strip=True)
        paragraphs = soup.find_all('p')
        text = ' '.join(para.get_text() for para in paragraphs)
        return text if text else "No significant content found."
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to scrape article text from {url}: {e}")
        return f"Unable to access content due to error: {str(e)}"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Fetch News Headlines from RSS Feed
def fetch_news_headlines(feed_url="http://feeds.bbci.co.uk/news/world/africa/rss.xml", max_headlines=5):
    headlines_by_source = {}
    try:
        feed = feedparser.parse(feed_url)
        if feed.entries:
            source = feed.feed.get('title', 'Unknown Source')
            headlines = []
            for entry in feed.entries[:max_headlines]:
                headlines.append({
                    'source': source,
                    'title': entry.get('title', 'No Title'),
                    'date': entry.get('published', 'N/A')
                })
            headlines_by_source[source] = headlines
        else:
            headlines_by_source["BBC News Africa"] = [{"source": "BBC News Africa", "title": "No headlines available.", "date": "N/A"}]
    except Exception as e:
        logger.error(f"Failed to fetch news headlines: {e}")
        headlines_by_source["BBC News Africa"] = [{"source": "BBC News Africa", "title": "Error fetching headlines.", "date": "N/A"}]
    return headlines_by_source

# Routes
@app.route('/')
def dashboard():
    city_conflicts = db.session.query(
        Prediction.location_name,
        func.sum(Prediction.conflict_level).label('total_conflict')
    ).group_by(Prediction.location_name).all()
    cities = [row.location_name for row in city_conflicts]
    conflict_levels = [row.total_conflict for row in city_conflicts]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']
    conflict_bar = json.dumps([go.Bar(
        x=cities,
        y=conflict_levels,
        name='Conflict Levels',
        marker=dict(color=colors[:len(cities)])
    )], cls=PlotlyJSONEncoder)

    response_times = {'Nairobi': 2.5, 'Mombasa': 3.0, 'Kisumu': 1.8, 'Eldoret': 2.2}
    response_bar = json.dumps([go.Bar(
        x=list(response_times.keys()),
        y=list(response_times.values()),
        name='Response Time (Hours)',
        marker=dict(color=colors[:len(response_times)])
    )], cls=PlotlyJSONEncoder)

    return render_template('dash.html', conflict_bar=conflict_bar, response_bar=response_bar)

@app.route('/predict_page')
def predict_page():
    locations = [
        ('Nairobi', -1.286389, 36.817223),
        ('Mombasa', -4.043477, 39.668206),
        ('Kisumu', -0.091702, 34.767956),
        ('Eldoret', 0.514277, 35.269780)
    ]
    extra_locations = {
        'Nairobi': [('CBD', -1.286389, 36.817223), ('Westlands', -1.238896, 36.813287), ('Kilimani', -1.288456, 36.823263)],
        'Mombasa': [('Old Town', -4.043477, 39.668206), ('Nyali', -4.031072, 39.718347), ('Mtwapa', -3.967537, 39.784317)],
        'Kisumu': [('Downtown', -0.091702, 34.767956), ('Kisumu West', -0.086276, 34.748763), ('Kisumu East', -0.085302, 34.783491)],
        'Eldoret': [('Market Area', 0.514277, 35.269780), ('Kapsaret', 0.528345, 35.268492), ('Town Centre', 0.520125, 35.269012)]
    }
    predictions = Prediction.query.all()

    map_data = [
        {
            'location_name': pred.location_name,
            'extra_location': pred.extra_location,
            'latitude': pred.latitude,
            'longitude': pred.longitude,
            'conflict_level': pred.conflict_level,
            'date': pred.date.strftime('%Y-%m-%d')
        } for pred in predictions
    ]
    map_data_json = json.dumps(map_data)

    data = db.session.query(
        Prediction.location_name,
        Prediction.date,
        func.sum(Prediction.conflict_level).label('total_conflict')
    ).group_by(Prediction.location_name, Prediction.date).order_by(Prediction.date).all()
    traces = []
    for location_name, group in groupby(data, key=lambda x: x.location_name):
        grouped_data = list(group)
        dates = [g.date for g in grouped_data]
        levels = [g.total_conflict for g in grouped_data]
        traces.append(go.Scatter(x=dates, y=levels, mode='lines+markers', name=location_name))
    timeline_chart_json = json.dumps(traces, cls=PlotlyJSONEncoder)

    city_conflicts = db.session.query(
        Prediction.location_name,
        func.sum(Prediction.conflict_level).label('total_conflict')
    ).group_by(Prediction.location_name).all()
    cities = [row.location_name for row in city_conflicts]
    conflict_levels = [row.total_conflict for row in city_conflicts]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']
    pie_chart_json = json.dumps({
        'labels': cities,
        'values': conflict_levels,
        'type': 'pie',
        'marker': {'colors': colors[:len(cities)]},
        'textinfo': 'label+percent',
        'hoverinfo': 'label+value+percent'
    }, cls=PlotlyJSONEncoder)

    tweets_by_user = {
        "amerix": [{"author": "amerix", "text": "Stay strong amid unrest.", "date": "Oct 25, 2023"}],
        "Kijanayamwingi7": [{"author": "Kijanayamwingi7", "text": "Nairobi protests today.", "date": "Oct 26, 2023"}],
        "Scrapfly_dev": [{"author": "Scrapfly_dev", "text": "Scraping test tweet.", "date": "Oct 27, 2023"}]
    }

    headlines_by_source = fetch_news_headlines(feed_url="http://feeds.bbci.co.uk/news/world/africa/rss.xml", max_headlines=5)

    prediction_result = session.pop('prediction_result', None)
    latest_prediction = session.get('latest_prediction', None)

    return render_template('index.html', locations=locations, extra_locations=extra_locations, 
                           timeline_chart=timeline_chart_json, pie_chart=pie_chart_json, 
                           tweets_by_user=tweets_by_user, headlines_by_source=headlines_by_source,
                           prediction_result=prediction_result, map_data_json=map_data_json,
                           latest_prediction=latest_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    link = request.form.get('link', '')
    combined_location = request.form.get('location', '')
    if not link or not combined_location:
        flash('Both link and location are required', 'error')
        return redirect(url_for('predict_page'))
    try:
        parts = combined_location.split('|')
        location_name = parts[0]
        latitude = float(parts[1])
        longitude = float(parts[2])
        extra_location = parts[3] if len(parts) > 3 else location_name

        article_text = scrape_article_text(link)
        cleaned_text = preprocess_text(article_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        result_label = "Civil Unrest" if prediction == 1 else "No Civil Unrest"

        today = datetime.today().date()
        today_pred = Prediction.query.filter_by(location_name=location_name, extra_location=extra_location, date=today).first()
        if today_pred:
            today_pred.conflict_level += int(prediction == 1)
        else:
            new_prediction = Prediction(
                location_name=location_name,
                extra_location=extra_location,
                latitude=latitude,
                longitude=longitude,
                conflict_level=int(prediction == 1),
                date=today
            )
            db.session.add(new_prediction)
        db.session.commit()

        session['prediction_result'] = {
            'label': result_label,
            'location': f"{location_name} - {extra_location}"
        }
        session['latest_prediction'] = {
            'latitude': latitude,
            'longitude': longitude,
            'conflict_level': int(prediction == 1),
            'location_name': location_name,
            'extra_location': extra_location
        }

        flash(f"Prediction for {location_name} - {extra_location} updated successfully!", 'success')
        return redirect(url_for('predict_page'))
    except Exception as e:
        logger.error(f"Error in predict route: {e}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('predict_page'))

@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    link = request.form.get('link', '')
    combined_location = request.form.get('location', '')
    if not link or not combined_location:
        logger.warning("Missing link or location in predict_ajax")
        return jsonify({'error': 'Both link and location are required'}), 400
    
    try:
        parts = combined_location.split('|')
        if len(parts) < 3:
            raise ValueError("Invalid location format")
        location_name = parts[0]
        latitude = float(parts[1])
        longitude = float(parts[2])
        extra_location = parts[3] if len(parts) > 3 else location_name

        logger.info(f"Processing prediction for {location_name} - {extra_location} with link {link}")
        article_text = scrape_article_text(link)
        if "Unable to access content" in article_text or not article_text.strip():
            logger.warning(f"Using fallback text for {link}")
            article_text = "Unable to scrape article content; assuming neutral context."

        cleaned_text = preprocess_text(article_text)
        vectorized_text = vectorizer.transform([cleaned_text])

        # Check feature compatibility
        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 30  # Adjust based on your model
        actual_features = vectorized_text.shape[1]
        if actual_features != expected_features:
            logger.warning(f"Feature mismatch: expected {expected_features}, got {actual_features}. Padding/truncating.")
            if actual_features < expected_features:
                # Pad with zeros
                padding = np.zeros((1, expected_features - actual_features))
                vectorized_text = np.hstack((vectorized_text.toarray(), padding))
                vectorized_text = np.sparse.csr_matrix(vectorized_text)
            else:
                # Truncate
                vectorized_text = vectorized_text[:, :expected_features]

        prediction = model.predict(vectorized_text)[0]
        result_label = "Civil Unrest" if prediction == 1 else "No Civil Unrest"

        today = datetime.today().date()
        today_pred = Prediction.query.filter_by(location_name=location_name, extra_location=extra_location, date=today).first()
        if today_pred:
            today_pred.conflict_level += int(prediction == 1)
        else:
            new_prediction = Prediction(
                location_name=location_name,
                extra_location=extra_location,
                latitude=latitude,
                longitude=longitude,
                conflict_level=int(prediction == 1),
                date=today
            )
            db.session.add(new_prediction)
        db.session.commit()

        city_conflicts = db.session.query(
            Prediction.location_name,
            func.sum(Prediction.conflict_level).label('total_conflict')
        ).group_by(Prediction.location_name).all()
        cities = [row.location_name for row in city_conflicts]
        conflict_levels = [row.total_conflict for row in city_conflicts]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']
        pie_data = {
            'labels': cities,
            'values': conflict_levels,
            'type': 'pie',
            'marker': {'colors': colors[:len(cities)]},
            'textinfo': 'label+percent',
            'hoverinfo': 'label+value+percent'
        }

        logger.info(f"Prediction successful for {location_name} - {extra_location}: {result_label}")
        return jsonify({
            'success': True,
            'prediction': {
                'label': result_label,
                'location': f"{location_name} - {extra_location}",
                'latitude': latitude,
                'longitude': longitude,
                'conflict_level': int(prediction == 1)
            },
            'pie_data': pie_data
        })
    except ValueError as ve:
        logger.error(f"ValueError in predict_ajax: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in predict_ajax: {e}")
        db.session.rollback()
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.check_password(password):
            login_user(admin)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard_admin'))
        else:
            flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/admin')
@login_required
def dashboard_admin():
    predictions = Prediction.query.all()
    threshold = 50
    today = datetime.today().date()
    recent_alerts = db.session.query(
        Prediction.location_name,
        Prediction.extra_location,
        Prediction.date,
        func.sum(Prediction.conflict_level).label('total_conflict')
    ).filter(Prediction.date >= today - timedelta(days=3)) \
     .group_by(Prediction.location_name, Prediction.extra_location, Prediction.date) \
     .having(func.sum(Prediction.conflict_level) >= threshold).all()
    timeline_data = db.session.query(
        Prediction.location_name,
        Prediction.date,
        func.sum(Prediction.conflict_level).label('total_conflict')
    ).group_by(Prediction.location_name, Prediction.date).order_by(Prediction.date).all()
    traces = []
    for location_name, group in groupby(timeline_data, key=lambda x: x.location_name):
        grouped_data = list(group)
        dates = [g.date for g in grouped_data]
        levels = [g.total_conflict for g in grouped_data]
        traces.append(go.Scatter(x=dates, y=levels, mode='lines+markers', name=location_name))
    timeline_chart_json = json.dumps(traces, cls=PlotlyJSONEncoder)
    return render_template('dashboard.html', predictions=predictions, recent_alerts=recent_alerts, timeline_chart_json=timeline_chart_json)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        create_default_admin()
        if Prediction.query.count() == 0:
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 1, 17)
            delta = timedelta(days=1)
            locations = [
                ('Nairobi', [('CBD', -1.286389, 36.817223), ('Westlands', -1.238896, 36.813287), ('Kilimani', -1.288456, 36.823263)]),
                ('Mombasa', [('Old Town', -4.043477, 39.668206), ('Nyali', -4.031072, 39.718347), ('Mtwapa', -3.967537, 39.784317)]),
                ('Kisumu', [('Downtown', -0.091702, 34.767956), ('Kisumu West', -0.086276, 34.748763), ('Kisumu East', -0.085302, 34.783491)]),
                ('Eldoret', [('Market Area', 0.514277, 35.269780), ('Kapsaret', 0.528345, 35.268492), ('Town Centre', 0.520125, 35.269012)])
            ]
            current_date = start_date
            while current_date <= end_date:
                for city_name, extra_locations in locations:
                    for extra_location, lat, lon in extra_locations:
                        conflict_level = random.randint(0, 100)
                        new_pred = Prediction(
                            location_name=city_name,
                            extra_location=extra_location,
                            latitude=lat,
                            longitude=lon,
                            conflict_level=conflict_level,
                            date=current_date
                        )
                        db.session.add(new_pred)
                current_date += delta
            db.session.commit()
    app.run(debug=True)