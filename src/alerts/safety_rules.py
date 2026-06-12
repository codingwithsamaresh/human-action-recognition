"""
Safety Rules

Maps action labels
to severity levels.
"""


class SafetyRules:

    def __init__(self):

        self.rules = {

            "Walking": "LOW",

            "Running": "LOW",

            "Sitting": "LOW",

            "Punching": "HIGH",

            "Falling": "CRITICAL",

            "TestAction": "LOW"
        }

    def get_severity(
        self,
        action
    ):

        return self.rules.get(
            action,
            "UNKNOWN"
        )

    def should_alert(
        self,
        action
    ):

        severity = self.get_severity(
            action
        )

        return severity in [
            "HIGH",
            "CRITICAL"
        ]