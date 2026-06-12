"""
Sound Alert

Plays a warning sound
for HIGH and CRITICAL events.
"""

import winsound


class SoundAlert:

    def __init__(
        self,
        frequency=1000,
        duration=500
    ):
        self.frequency = frequency
        self.duration = duration

    def trigger(
        self,
        severity
    ):
        """
        Args:
            severity:
                LOW
                HIGH
                CRITICAL
        """

        if severity == "HIGH":

            winsound.Beep(
                self.frequency,
                self.duration
            )

        elif severity == "CRITICAL":

            for _ in range(3):

                winsound.Beep(
                    self.frequency,
                    self.duration
                )