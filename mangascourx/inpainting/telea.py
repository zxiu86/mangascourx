import numpy as np

from .base import Inpainter
from ..core.priority_queue import PriorityQueue


class TeleaInpainter(Inpainter):

    def __init__(self, radius=5):

        self.radius = radius

    def run(self, image, mask):

        result = image.astype(np.float32).copy()

        h, w = mask.shape

        dist = np.full(
            (h, w),
            np.inf,
            dtype=np.float32
        )

        pq = PriorityQueue(h * w)

        boundary = self._find_boundary(mask)

        for y, x in boundary:

            dist[y, x] = 0.0

            pq.push(
                0.0,
                y,
                x
            )

        while not pq.empty():

            d, y, x = pq.pop()

            self._propagate(
                result,
                mask,
                dist,
                pq,
                y,
                x
            )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)