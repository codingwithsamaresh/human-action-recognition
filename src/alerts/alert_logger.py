import json
import os
from datetime import datetime


class AlertLogger:
    def __init__(
        self,
        log_file="outputs/predictions/alerts.json"
    ):
        self.log_file = log_file

        os.makedirs(
            os.path.dirname(self.log_file),
            exist_ok=True
        )

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def log_alert(
        self,
        person_id,
        action,
        confidence,
        severity
    ):
        alert = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "person_id": person_id,
            "action": action,
            "confidence": float(confidence),
            "severity": severity
        }

        with open(self.log_file, "r") as f:
            data = json.load(f)

        data.append(alert)

        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=4)

        return alert