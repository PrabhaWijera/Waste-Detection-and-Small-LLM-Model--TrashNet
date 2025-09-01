import smtplib
from email.mime.text import MIMEText

from_email = "prabhashwlive2001@gmail.com"
password = "shaaumbtribzkecz"  # App Password, no spaces
to_email = "managementdevelopers@gmail.com"

msg = MIMEText("Test email from Python SMTP_SSL")
msg["Subject"] = "Test Email"
msg["From"] = from_email
msg["To"] = to_email

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
    print("Email sent successfully!")
except Exception as e:
    print("Email failed:", e)
