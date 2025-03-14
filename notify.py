from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up APScheduler
scheduler = BackgroundScheduler()
scheduler.start()

def send_email():
    # Use smtplib to send an email
    sender_email = "your-email@example.com"
    receiver_email = "receiver-email@example.com"
    password = "your-email-password"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Daily Alert"
    
    body = "This is the daily conflict alert summary."
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Schedule email every day at a specific time (e.g., at 9:00 AM)
scheduler.add_job(
    func=send_email,
    trigger=IntervalTrigger(hours=24),
    id='email_job',
    name='Send daily email',
    replace_existing=True
)
