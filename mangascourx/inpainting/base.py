"""
inpainting/base.py — Abstract base class for all inpainting algorithms.

NOTE: This is DIFFERENT from detection/base.py (which has BaseDetector).
      This file provides the Inpainter contract.

Bugs fixed:
  5. This file didn't exist at all — telea.py and coherence.py both import
     `from .base import Inpainter` which caused ImportError.
  6. _find_boundary() and _propagate() were called by telea.py via self.*
     but were never defined anywhere. They live here so every Inpainter
     subclass inherits them automatically.
"""
from __future__ import annotations

import abc
import numpy as np
from numpy.typing import NDArray


class Inpainter(abc.ABC):
    """
    Abstract base for image inpainting algorithms.

    Contract
    --------
    Subclasses must implement `run(image, mask) -> np.ndarray`.
    `_find_boundary` and `_propagate` are utility methods available to all
    subclasses (used by TeleaInpainter).
    """

    @abc.abstractmethod
    def run(
        self,
        image: NDArray[np.uint8],
        mask: NDArray,
    ) -> NDArray[np.uint8]:
        """
        Fill the masked holes in `image`.

        Args:
            image : BGR uint8 array (H×W×C).
            mask  : 2-D array — non-zero pixels = holes to fill.

        Returns:
            Inpainted uint8 image, same shape as input.
        """
        ...

    # ── Bug 6 Fix: _find_boundary ────────────────────────────────────────────
    @staticmethod
    def _find_boundary(mask: NDArray) -> list[tuple[int, int]]:
        """
        Return every pixel that is INSIDE the hole but is adjacent
        (4-connected) to at least one known (non-hole) pixel.

        This is the starting frontier for Fast-Marching inpainting.

        Args:
            mask: 2-D array; non-zero = hole.

        Returns:
            List of (y, x) tuples.
        """
        h, w = mask.shape
        boundary: list[tuple[int, int]] = []
        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                        boundary.append((y, x))
                        break
        return boundary

    # ── Bug 6 Fix: _propagate ────────────────────────────────────────────────
    def _propagate(
        self,
        result: NDArray[np.float32],
        mask: NDArray,
        dist: NDArray[np.float32],
        pq,
        y: int,
        x: int,
    ) -> None:
        """
        Fill result[y, x] with a distance-weighted average of its known
        8-connected neighbours, then push newly reachable masked neighbours
        onto the min-heap priority queue.

        This implements one step of the Telea / Fast-Marching propagation.

        Args:
            result : Float image being reconstructed (modified in-place).
            mask   : Hole mask (non-zero = hole).
            dist   : Distance map (modified in-place for newly reached pixels).
            pq     : PriorityQueue instance — push(distance, y, x).
            y, x   : Coordinates of the pixel being filled right now.
        """
        h, w = mask.shape
        ndim = result.ndim
        c = result.shape[2] if ndim == 3 else 1

        # Weighted average from all known 8-neighbours
        accum = np.zeros(c, dtype=np.float64)
        w_total = 0.0

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if mask[ny, nx]:
                continue                              # still a hole → skip

            eucl = float(dy * dy + dx * dx) ** 0.5
            # Telea weight: closer pixel + smaller existing distance is better
            wv = 1.0 / (eucl * (1.0 + dist[ny, nx]) + 1e-8)

            if ndim == 3:
                accum += wv * result[ny, nx].astype(np.float64)
            else:
                accum[0] += wv * float(result[ny, nx])
            w_total += wv

        if w_total > 0.0:
            filled = accum / w_total
            if ndim == 3:
                result[y, x] = filled.astype(np.float32)
            else:
                result[y, x] = float(filled[0])

        # Propagate distance to 4-connected masked neighbours
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not mask[ny, nx]:
                continue                              # already known

            new_dist = dist[y, x] + 1.0
            if new_dist < dist[ny, nx]:
                dist[ny, nx] = new_dist
                pq.push(new_dist, ny, nx)
