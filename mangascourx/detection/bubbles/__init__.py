"""Bubble detection — contour-based speech bubble segmentation."""
from .contours import detect_bubbles
from .morphology import clean_noise

__all__ = ["detect_bubbles", "clean_noise"]
