"""
Multi Person Frame Buffers

Maintains one frame buffer
for each tracked person.
"""

from src.inference.frame_buffer import (
    FrameBuffer
)


class MultiPersonBuffer:

    def __init__(
        self,
        sequence_length=16
    ):

        self.sequence_length = (
            sequence_length
        )

        self.buffers = {}

    def update(
        self,
        track_id,
        roi
    ):

        if track_id not in self.buffers:

            self.buffers[
                track_id
            ] = FrameBuffer(
                maxlen=self.sequence_length
            )

        self.buffers[
            track_id
        ].add(
            roi
        )

    def ready(
        self,
        track_id
    ):

        if track_id not in self.buffers:
            return False

        return self.buffers[
            track_id
        ].is_full()

    def get_sequence(
        self,
        track_id
    ):

        return self.buffers[
            track_id
        ].get_sequence()

    def remove(
        self,
        track_id
    ):

        if track_id in self.buffers:

            del self.buffers[
                track_id
            ]

    def get_active_ids(
        self
    ):

        return list(
            self.buffers.keys()
        )