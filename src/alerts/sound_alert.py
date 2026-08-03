"""
Sound Alert

Plays warning sounds for
HIGH and CRITICAL events.

Works on:
- Windows
- Linux
- Google Colab (prints warning if sound is unavailable)
"""

try:
    import winsound

    WINDOWS_SOUND = True

except ImportError:

    WINDOWS_SOUND = False


class SoundAlert:
    """
    Sound alert utility.

    HIGH:
        One beep

    CRITICAL:
        Three beeps
    """

    def __init__(
        self,
        frequency=1000,
        duration=500
    ):
        self.frequency = frequency
        self.duration = duration

    def _beep(self):
        """
        Produce one beep.

        On Windows:
            Uses winsound.

        On Linux/Colab:
            Falls back to console message.
        """

        if WINDOWS_SOUND:

            winsound.Beep(
                self.frequency,
                self.duration
            )

        else:

            print("\a", end="", flush=True)

    def trigger(
        self,
        severity
    ):
        """
        Trigger sound alert.

        Parameters
        ----------
        severity : str
            LOW
            HIGH
            CRITICAL
        """

        severity = severity.upper()

        if severity == "LOW":
            return

        elif severity == "HIGH":

            self._beep()

        elif severity == "CRITICAL":

            for _ in range(3):

                self._beep()

        else:

            print(
                f"Unknown alert severity: {severity}"
            )