"""
Alert Manager

Receives action predictions,
checks severity rules,
logs incidents,
and decides whether
an alert should be triggered.
"""

from src.alerts.safety_rules import SafetyRules
from src.alerts.alert_logger import AlertLogger


class AlertManager:

    def __init__(self):

        self.rules = SafetyRules()
        self.logger = AlertLogger()

    def process_prediction(
        self,
        track_id,
        action,
        confidence
    ):
        """
        Process a single action prediction.

        Parameters
        ----------
        track_id : int
            Unique tracked person ID.

        action : str
            Predicted action label.

        confidence : float
            Model confidence score.

        Returns
        -------
        dict
            {
                "track_id": int,
                "action": str,
                "confidence": float,
                "severity": str,
                "alert": bool
            }
        """

        severity = self.rules.get_severity(
            action
        )

        alert = self.rules.should_alert(
            action
        )

        result = {
            "track_id": track_id,
            "action": action,
            "confidence": float(confidence),
            "severity": severity,
            "alert": alert
        }

        # Log only alert-worthy events
        if alert:

            self.logger.log_alert(
                person_id=track_id,
                action=action,
                confidence=confidence,
                severity=severity
            )

        return result