import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

class AlertSystem:
    """Automated email alerts for hazards."""
    def __init__(self, sender_email, receiver_email, smtp_server='smtp.example.com', port=587, password='password'):
        self.sender = sender_email
        self.receiver = receiver_email
        self.smtp_server = smtp_server
        self.port = port
        self.password = password
        self.last_alert_time = 0
        self.cooldown = 2  # Seconds between alerts

    def send_alert(self, hazard_info):
        """Send email alert if cooldown passed."""
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown:
            return
        self.last_alert_time = current_time

        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = self.receiver
        msg['Subject'] = 'Hazard Alert: Proximity Detected'
        body = f"Hazard detected: {hazard_info}"
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(self.smtp_server, self.port) as server:
            server.starttls()
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.receiver, msg.as_string())