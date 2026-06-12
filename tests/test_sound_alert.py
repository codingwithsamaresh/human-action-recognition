from src.alerts.sound_alert import (
    SoundAlert
)


def main():

    alert = SoundAlert()

    print(
        "Testing HIGH alert..."
    )

    alert.trigger(
        "HIGH"
    )

    print(
        "Testing CRITICAL alert..."
    )

    alert.trigger(
        "CRITICAL"
    )

    print(
        "Done."
    )


if __name__ == "__main__":
    main()