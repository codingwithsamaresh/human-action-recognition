from src.alerts.alert_manager import (
    AlertManager
)


def main():

    manager = AlertManager()

    result1 = (
        manager.process_prediction(
            track_id=0,
            action="Walking",
            confidence=0.95
        )
    )

    result2 = (
        manager.process_prediction(
            track_id=1,
            action="Falling",
            confidence=0.99
        )
    )

    print(result1)
    print(result2)


if __name__ == "__main__":
    main()