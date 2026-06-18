"""
Central detection orchestrator.

Bugs fixed vs original detection.py:
  1. from .masks  → from .mask          (wrong module name)
  2. class Detector → DetectionOrchestrator  (name mismatch with text_remove.py)
  3. .run() method added                (text_remove.py calls .run(), original only had .detect())
  4. detect_text_regions(**self.mser_params) → (mser_params=self.mser_params)
     (function signature is (image, mser_params=None, ...) not (**kwargs))
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .bubbles.morphology import clean_noise
from .bubbles.contours import detect_bubbles
from .text.mser import detect_text_regions
from .text.craft_adapter import CRAFTDetector
from .mask import process_masks          # ← BUG FIX 1: was ".masks"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _generate_bubble_mask(
    image: NDArray[np.uint8],
    *,
    block_size: int = 31,
    c: int = 5,
    morph_open: int = 3,
    morph_close: int = 5,
) -> NDArray[np.uint8]:
    """Adaptive-threshold bubble candidate mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c,
    )
    return clean_noise(mask, open_kernel_size=morph_open, close_kernel_size=morph_close)


# ── BUG FIX 2: renamed from "Detector" → "DetectionOrchestrator" ─────────────
class DetectionOrchestrator:
    """
    Unified detector for text and bubbles.

    Parameters
    ----------
    craft_model_path : str or None
    device : str
    mser_params : dict or None   — passed as mser_params= to detect_text_regions
    craft_params : dict or None
    bubble_params : dict or None
    merge_priority : list[str]
    final_cleanup : bool
    fallback_min_boxes : int
    """

    def __init__(
        self,
        craft_model_path: Optional[str] = None,
        device: str = "cpu",
        mser_params: Optional[dict] = None,
        craft_params: Optional[dict] = None,
        bubble_params: Optional[dict] = None,
        merge_priority: Optional[List[str]] = None,
        final_cleanup: bool = True,
        fallback_min_boxes: int = 3,
    ) -> None:
        self.craft_model_path = craft_model_path
        self.device = device
        self.mser_params = mser_params or {}
        self.craft_params = craft_params or {}
        self.bubble_params = bubble_params or {}
        self.merge_priority = merge_priority or ["text", "bubbles"]
        self.final_cleanup = final_cleanup
        self.fallback_min_boxes = fallback_min_boxes

        self._craft_detector: Optional[CRAFTDetector] = None
        if craft_model_path:
            try:
                self._craft_detector = CRAFTDetector(
                    craft_model_path, device=device, **self.craft_params
                )
            except Exception as e:
                logger.warning(f"Failed to load CRAFT model: {e}")

    # ------------------------------------------------------------------
    def _detect_text(
        self, image: NDArray[np.uint8]
    ) -> Tuple[List[Tuple[int, int, int, int]], NDArray[np.uint8]]:
        # ── BUG FIX 4: was detect_text_regions(image, **self.mser_params)
        #    The function signature is (image, mser_params=None, filter_params=None)
        #    Spreading the dict with ** passes wrong keyword arguments. ──────────
        boxes = detect_text_regions(
            image,
            mser_params=self.mser_params or None,
        )

        if self._craft_detector is not None and len(boxes) < self.fallback_min_boxes:
            logger.info(f"MSER found {len(boxes)} boxes — falling back to CRAFT.")
            try:
                boxes = self._craft_detector.detect(image)
            except Exception as e:
                logger.error(f"CRAFT detection failed: {e}")

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
        return boxes, mask

    # ------------------------------------------------------------------
    def _detect_bubbles(
        self, image: NDArray[np.uint8]
    ) -> Tuple[List[Tuple[int, int, int, int]], NDArray[np.uint8]]:
        raw_mask = _generate_bubble_mask(image, **self.bubble_params)
        bubble_data = detect_bubbles(raw_mask)
        contours = bubble_data["contours"]
        bounding_rects = bubble_data["bounding_rects"]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=-1)
        return bounding_rects, mask

    # ------------------------------------------------------------------
    def detect(
        self,
        image: NDArray[np.uint8],
        *,
        enable_text: bool = True,
        enable_bubbles: bool = True,
    ) -> Dict:
        """Core detection method — returns dict with 'mask' and metadata."""
        masks_dict: Dict[str, NDArray[np.uint8]] = {}
        text_boxes: List = []
        bubble_boxes: List = []

        if enable_text:
            text_boxes, text_mask = self._detect_text(image)
            masks_dict["text"] = text_mask

        if enable_bubbles:
            bubble_boxes, bubble_mask = self._detect_bubbles(image)
            masks_dict["bubbles"] = bubble_mask

        if not masks_dict:
            final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            final_mask = process_masks(
                masks_dict,
                priority=self.merge_priority,
                cleanup=self.final_cleanup,
                return_labeled=False,
            )

        result: Dict = {"mask": final_mask, "priority": self.merge_priority}
        if enable_text:
            result["text_boxes"] = text_boxes
            result["text_mask"] = masks_dict.get("text")
        if enable_bubbles:
            result["bubble_boxes"] = bubble_boxes
            result["bubble_mask"] = masks_dict.get("bubbles")
        return result

    # ── BUG FIX 3: text_remove.py calls .run() not .detect() ────────────────
    def run(
        self,
        image: NDArray[np.uint8],
        enable_text: bool = True,
        enable_bubbles: bool = True,
    ) -> Dict:
        """Alias for detect() — the pipeline calls .run() by convention."""
        return self.detect(image, enable_text=enable_text, enable_bubbles=enable_bubbles)


# Backward-compat alias (old code that used "Detector" directly still works)
Detector = DetectionOrchestrator
