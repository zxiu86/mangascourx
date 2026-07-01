"""
Fast text detection using MSER (Maximally Stable Extremal Regions).

This module provides a lightweight, non-AI text detector that locates
candidate text regions by leveraging MSER and filtering candidates
based on geometric properties (area, aspect ratio, solidity, etc.).

Typical usage:
    from text.mser import detect_text_regions, draw_regions

    regions = detect_text_regions(image)
    result  = draw_regions(image, regions)
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Default parameters (can be tuned for specific use cases)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # MSER parameters
    "delta": 5,
    "min_area": 60,
    "max_area": 14400,
    "max_variation": 0.25,
    "min_diversity": 0.2,
    # Geometric filters
    "aspect_ratio_min": 0.3,
    "aspect_ratio_max": 10.0,
    "solidity_min": 0.3,
    "extent_min": 0.2,
    "min_width": 8,
    "min_height": 8,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert an image to grayscale if it is not already."""
    if image.ndim == 3:
        # Assume BGR if 3 channels
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Assume single channel already if 3rd dim == 1
        if image.shape[2] == 1:
            return image[:, :, 0]
    if image.ndim == 2:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


# ---------------------------------------------------------------------------
# MSER‑based text detection
# ---------------------------------------------------------------------------

def create_mser(
    delta: int = 5,
    min_area: int = 60,
    max_area: int = 14400,
    max_variation: float = 0.25,
    min_diversity: float = 0.2,
) -> cv2.MSER:
    """
    Create an OpenCV MSER detector with custom parameters.

    Args:
        delta: Delta, used in MSER stability calculation.
        min_area: Minimum area of a detected region (in pixels).
        max_area: Maximum area of a detected region.
        max_variation: Maximum area variation between stable regions.
        min_diversity: Minimum diversity to cut off the tree.

    Returns:
        Configured MSER object.
    """
    # BUG FIX: cv2.MSER_create() dropped the underscore-prefixed kwarg
    # names (_delta, _min_area, ...) starting around OpenCV 4.5.1+.
    # Modern OpenCV (anything matching this package's own
    # "opencv-python-headless>=4.5.0" requirement) raises:
    #   TypeError: '_delta' is an invalid keyword argument for MSER_create()
    # Try the modern names first, fall back to the legacy underscored
    # names so this still works on old OpenCV 3.x/early-4.x wheels.
    try:
        return cv2.MSER_create(
            delta=delta,
            min_area=min_area,
            max_area=max_area,
            max_variation=max_variation,
            min_diversity=min_diversity,
        )
    except TypeError:
        return cv2.MSER_create(
            _delta=delta,
            _min_area=min_area,
            _max_area=max_area,
            _max_variation=max_variation,
            _min_diversity=min_diversity,
        )


def _is_valid_region(
    bbox: tuple[int, int, int, int],
    region_mask: NDArray[np.uint8],
    params: dict,
) -> bool:
    """
    Check whether a candidate bounding box and its mask satisfy
    geometric constraints.

    Args:
        bbox: (x, y, w, h) of the region.
        region_mask: Binary mask of the region (single‑channel, 0/255).
        params: Dictionary of filter thresholds.

    Returns:
        True if the region passes all filters.
    """
    x, y, w, h = bbox
    if w < params["min_width"] or h < params["min_height"]:
        return False

    aspect_ratio = w / h if h != 0 else 0
    if not (params["aspect_ratio_min"] <= aspect_ratio <= params["aspect_ratio_max"]):
        return False

    # Solidity = region area / convex hull area
    region_area = cv2.countNonZero(region_mask)
    if region_area == 0:
        return False
    hull = cv2.convexHull(cv2.findNonZero(region_mask))
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    solidity = region_area / hull_area
    if solidity < params["solidity_min"]:
        return False

    # Extent = region area / bounding box area
    bbox_area = w * h
    extent = region_area / bbox_area if bbox_area != 0 else 0
    if extent < params["extent_min"]:
        return False

    return True


def detect_text_regions(
    image: NDArray[np.uint8],
    mser_params: dict | None = None,
    filter_params: dict | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Detect text candidate regions in an image using MSER.

    Args:
        image: Input image (BGR, grayscale, or any uint8 format).
        mser_params: Overrides for MSER constructor (delta, min_area, etc.).
            If None, DEFAULT_PARAMS values are used.
        filter_params: Overrides for geometric filtering thresholds.
            If None, DEFAULT_PARAMS values are used.

    Returns:
        List of bounding boxes as (x, y, width, height) tuples.
        Only regions that pass geometric filters are returned.
    """
    # Merge default parameters with user overrides
    _mser_params = DEFAULT_PARAMS.copy()
    if mser_params:
        _mser_params.update(mser_params)

    _filter_params = DEFAULT_PARAMS.copy()
    if filter_params:
        _filter_params.update(filter_params)

    gray = _to_gray(image)

    mser = create_mser(
        delta=_mser_params["delta"],
        min_area=_mser_params["min_area"],
        max_area=_mser_params["max_area"],
        max_variation=_mser_params["max_variation"],
        min_diversity=_mser_params["min_diversity"],
    )

    # MSER returns two lists: regions (lists of points) and their bboxes
    regions, bboxes = mser.detectRegions(gray)

    if regions is None or bboxes is None or len(regions) == 0:
        return []

    valid_boxes = []
    for region_pts, bbox in zip(regions, bboxes):
        # Construct a binary mask for the region to compute solidity/extent
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [region_pts.reshape(-1, 1, 2)], 255)

        if _is_valid_region(bbox, mask, _filter_params):
            valid_boxes.append(tuple(bbox))

    return valid_boxes


# ---------------------------------------------------------------------------
# Grouping overlapping boxes (simple NMS)
# ---------------------------------------------------------------------------

def non_max_suppression(
    boxes: list[tuple[int, int, int, int]],
    overlap_threshold: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """
    Apply simple non‑maximum suppression to a list of bounding boxes.

    Boxes are merged greedily: if two boxes overlap more than the
    threshold, they are combined into one.

    Args:
        boxes: List of (x, y, w, h) bounding boxes.
        overlap_threshold: IOU threshold above which boxes are merged.

    Returns:
        List of suppressed bounding boxes.
    """
    if not boxes:
        return []

    # Convert to float for calculations
    boxes_float = np.array(boxes, dtype=np.float32)
    x1 = boxes_float[:, 0]
    y1 = boxes_float[:, 1]
    x2 = boxes_float[:, 0] + boxes_float[:, 2]
    y2 = boxes_float[:, 1] + boxes_float[:, 3]
    areas = boxes_float[:, 2] * boxes_float[:, 3]

    order = areas.argsort()[::-1]  # sort by area descending
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= overlap_threshold)[0]
        order = order[inds + 1]

    return [boxes[i] for i in keep]


# ---------------------------------------------------------------------------
# Drawing utility
# ---------------------------------------------------------------------------

def draw_regions(
    image: NDArray[np.uint8],
    regions: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    inplace: bool = False,
) -> NDArray[np.uint8]:
    """
    Draw bounding boxes of detected regions on an image.

    Args:
        image: Input image (uint8, BGR or grayscale).
        regions: List of (x, y, w, h) tuples.
        color: BGR color of the rectangles.
        thickness: Line thickness.
        inplace: If True, draw on the input image directly.

    Returns:
        Image with rectangles drawn.
    """
    out = image if inplace else image.copy()
    for x, y, w, h in regions:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out
