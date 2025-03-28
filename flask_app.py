import os
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

# Load the model, vectorizer, and selector with error handling
try:
    with open('civil_unrest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        logger.info("Model loaded successfully!")
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
        logger.info("Vectorizer loaded successfully!")
    with open('feature_selector.pkl', 'rb') as selector_file:
        selector = pickle.load(selector_file)
        logger.info("Feature selector loaded successfully!")
except FileNotFoundError as e:
    logger.error(f"Failed to load model, vectorizer, or selector files: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading model, vectorizer, or selector: {e}")
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

class UserLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('admin.id'), nullable=True)
    username = db.Column(db.String(100), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)

    def __repr__(self):
        return f"<UserLog {self.username} - {self.action} at {self.timestamp}>"

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

# Helper Functions
def log_user_action(action, details=None, user=None):
    ip_address = request.remote_addr
    username = user.username if user else "anonymous"
    user_id = user.id if user else None
    
    user_log = UserLog(
        user_id=user_id,
        username=username,
        action=action,
        details=details,
        ip_address=ip_address
    )
    db.session.add(user_log)
    db.session.commit()
    
    logger.info(f"User Action: {action} by {username} (IP: {ip_address}) - Details: {details or 'None'}")

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

def send_sms(message):
    url = "https://api2.tiaraconnect.io/api/messaging/sendsms"
    headers = {
        "Authorization": f"Bearer {os.getenv('API_KEY')}",
        "Content-Type": "application/json"
    }
    recipients = os.getenv("SMS_TO", "").split(",")

    responses = []
    for recipient in recipients:
        data = {
            "from": os.getenv("SMS_FROM", "TIARA"),
            "to": recipient,
            "message": message,
        }
        response = requests.post(url, json=data, headers=headers)
        responses.append({
            "to": recipient,
            "status_code": response.status_code,
            "response": response.json()
        })
    logger.info(f"SMS sent: {responses}")
    return responses

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
    log_user_action("dashboard_access", "Accessed public dashboard")
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
    log_user_action("predict_page_access", "Accessed predict page", user=current_user if current_user.is_authenticated else None)
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
        log_user_action("prediction_attempt", "Missing link or location", user=current_user if current_user.is_authenticated else None)
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
        vectorized_text = selector.transform(vectorized_text)
        prediction = model.predict(vectorized_text)[0]
        prediction_proba = model.predict_proba(vectorized_text)[0]
        result_label = "Civil Unrest" if prediction == 1 else "No Civil Unrest"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

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

        details = f"Prediction for {location_name} - {extra_location}: {result_label} (Confidence: {confidence * 100:.2f}%)"
        log_user_action("prediction", details, user=current_user if current_user.is_authenticated else None)

        session['prediction_result'] = {
            'label': result_label,
            'location': f"{location_name} - {extra_location}",
            'confidence': f"{confidence * 100:.2f}%"
        }
        session['latest_prediction'] = {
            'latitude': latitude,
            'longitude': longitude,
            'conflict_level': int(prediction == 1),
            'location_name': location_name,
            'extra_location': extra_location
        }

        flash(f"Prediction for {location_name} - {extra_location}: {result_label} (Confidence: {confidence * 100:.2f}%)", 'success')
        return redirect(url_for('predict_page'))
    except Exception as e:
        log_user_action("prediction_error", f"Error: {str(e)}", user=current_user if current_user.is_authenticated else None)
        logger.error(f"Error in predict route: {e}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('predict_page'))

@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    link = request.form.get('link', '')
    combined_location = request.form.get('location', '')
    if not link or not combined_location:
        log_user_action("prediction_attempt_ajax", "Missing link or location", user=current_user if current_user.is_authenticated else None)
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
        vectorized_text = selector.transform(vectorized_text)

        # Check feature compatibility
        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else vectorizer.get_feature_names_out().size
        actual_features = vectorized_text.shape[1]
        if actual_features != expected_features:
            logger.warning(f"Feature mismatch: expected {expected_features}, got {actual_features}. Padding/truncating.")
            if actual_features < expected_features:
                padding = np.zeros((1, expected_features - actual_features))
                vectorized_text = np.hstack((vectorized_text.toarray(), padding))
                vectorized_text = np.sparse.csr_matrix(vectorized_text)
            else:
                vectorized_text = vectorized_text[:, :expected_features]

        prediction = model.predict(vectorized_text)[0]
        prediction_proba = model.predict_proba(vectorized_text)[0]
        result_label = "Civil Unrest" if prediction == 1 else "No Civil Unrest"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

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

        message = f"Prediction for {location_name} - {extra_location} updated successfully!: {result_label} (Confidence: {confidence * 100:.2f}%)"
        send_sms(message)

        details = f"Prediction for {location_name} - {extra_location}: {result_label} (Confidence: {confidence * 100:.2f}%)"
        log_user_action("prediction_ajax", details, user=current_user if current_user.is_authenticated else None)
        logger.info(f"Prediction successful for {location_name} - {extra_location}: {result_label}")
        return jsonify({
            'success': True,
            'prediction': {
                'label': result_label,
                'location': f"{location_name} - {extra_location}",
                'latitude': latitude,
                'longitude': longitude,
                'conflict_level': int(prediction == 1),
                'confidence': confidence * 100
            },
            'pie_data': pie_data
        })
    except ValueError as ve:
        log_user_action("prediction_error_ajax", f"ValueError: {str(ve)}", user=current_user if current_user.is_authenticated else None)
        logger.error(f"ValueError in predict_ajax: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        log_user_action("prediction_error_ajax", f"Unexpected error: {str(e)}", user=current_user if current_user.is_authenticated else None)
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
            log_user_action("login", f"Successful login for {username}", user=admin)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard_admin'))
        else:
            log_user_action("login_attempt", f"Failed login attempt for {username}")
            flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/admin')
@login_required
def dashboard_admin():
    log_user_action("admin_access", "Accessed admin dashboard", user=current_user)
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

@app.route('/admin/logs')
@login_required
def view_logs():
    log_user_action("logs_access", "Viewed system logs", user=current_user)
    logs = UserLog.query.order_by(UserLog.timestamp.desc()).limit(100).all()
    return render_template('logs.html', logs=logs)

@app.route('/logout')
@login_required
def logout():
    log_user_action("logout", f"User {current_user.username} logged out", user=current_user)
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