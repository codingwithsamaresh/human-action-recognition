"""
Email Alert

Sends email notifications
for HIGH and CRITICAL events.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailAlert:

    def __init__(
        self,
        smtp_server,
        smtp_port,
        sender_email,
        sender_password,
        receiver_email
    ):

        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

        self.sender_email = sender_email
        self.sender_password = sender_password

        self.receiver_email = receiver_email

    def send_alert(
        self,
        track_id,
        action,
        confidence,
        severity
    ):
        """
        Send alert email.
        """

        subject = (
            f"[{severity}] "
            f"Action Detected"
        )

        body = (
            f"Track ID: {track_id}\n"
            f"Action: {action}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Severity: {severity}"
        )

        message = MIMEMultipart()

        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = subject

        message.attach(
            MIMEText(
                body,
                "plain"
            )
        )

        try:

            with smtplib.SMTP(
                self.smtp_server,
                self.smtp_port
            ) as server:

                server.starttls()

                server.login(
                    self.sender_email,
                    self.sender_password
                )

                server.sendmail(
                    self.sender_email,
                    self.receiver_email,
                    message.as_string()
                )

            print(
                "Email alert sent."
            )

        except Exception as e:

            print(
                "Email alert failed:",
                e
            )