from src.alerts.email_alert import (
    EmailAlert
)


def test_email_alert():

    alert = EmailAlert(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        sender_email="bcsesamareshkoley01@gmail.com",
        sender_password="cevc rrcn dvoc reon",
        receiver_email="samareshkoley007@gmail.com"
    )

    alert.send_alert(
        track_id=0,
        action="Falling",
        confidence=0.99,
        severity="CRITICAL"
    )


if __name__ == "__main__":
    test_email_alert()