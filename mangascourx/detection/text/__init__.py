"""Text detection subpackage — MSER, SWT, CRAFT."""
from .mser import detect_text_regions, create_mser, non_max_suppression, draw_regions
from .craft_adapter import CRAFTDetector

__all__ = [
    "detect_text_regions", "create_mser", "non_max_suppression",
    "draw_regions", "CRAFTDetector",
]
