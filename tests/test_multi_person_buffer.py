from src.inference.multi_person_buffer import (
    MultiPersonBuffer
)

import numpy as np


buffer = MultiPersonBuffer(
    sequence_length=4
)

dummy = np.zeros(
    (224,224,3),
    dtype=np.uint8
)

for _ in range(4):
    buffer.update(0, dummy)

print(
    buffer.ready(0)
)

print(
    len(
        buffer.get_sequence(0)
    )
)