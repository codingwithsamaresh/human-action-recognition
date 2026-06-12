from src.alerts.safety_rules import (
    SafetyRules
)


def main():

    rules = SafetyRules()

    print(
        rules.get_severity(
            "Walking"
        )
    )

    print(
        rules.get_severity(
            "Falling"
        )
    )

    print(
        rules.should_alert(
            "Walking"
        )
    )

    print(
        rules.should_alert(
            "Falling"
        )
    )


if __name__ == "__main__":
    main()