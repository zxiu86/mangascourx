"""
Contour detection and processing tools for bubble masks.
All functions expect binary masks (uint8, 0 and 255).
"""
from __future__ import annotations
import cv2
import numpy as np
from numpy.typing import NDArray


def _ensure_binary_mask(mask: NDArray) -> NDArray[np.uint8]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def find_contours(mask: NDArray[np.uint8], mode: int = cv2.RETR_EXTERNAL,
                  method: int = cv2.CHAIN_APPROX_SIMPLE):
    mask = _ensure_binary_mask(mask)
    contours, hierarchy = cv2.findContours(mask.copy(), mode, method)
    return contours, hierarchy


def filter_contours_by_area(contours, min_area: float = 0.0,
                            max_area: float = float("inf")):
    if min_area == 0.0 and max_area == float("inf"):
        return contours
    return [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]


def get_convex_hulls(contours) -> list:
    return [cv2.convexHull(c) for c in contours if cv2.convexHull(c) is not None]


def get_bounding_rects(contours) -> list:
    return [cv2.boundingRect(c) for c in contours]


def get_bounding_circles(contours) -> list:
    result = []
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        result.append(((x, y), radius))
    return result


def draw_contours(image: NDArray[np.uint8], contours,
                  color=(0, 255, 0), thickness: int = 2,
                  inplace: bool = False) -> NDArray[np.uint8]:
    out = image if inplace else image.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def draw_convex_hulls(image: NDArray[np.uint8], hulls,
                      color=(255, 0, 0), thickness: int = 2,
                      inplace: bool = False) -> NDArray[np.uint8]:
    out = image if inplace else image.copy()
    cv2.drawContours(out, hulls, -1, color, thickness)
    return out


def draw_bounding_rects(image: NDArray[np.uint8], rects,
                        color=(0, 0, 255), thickness: int = 2,
                        inplace: bool = False) -> NDArray[np.uint8]:
    out = image if inplace else image.copy()
    for x, y, w, h in rects:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_bounding_circles(image: NDArray[np.uint8], circles,
                          color=(255, 255, 0), thickness: int = 2,
                          inplace: bool = False) -> NDArray[np.uint8]:
    out = image if inplace else image.copy()
    for (cx, cy), radius in circles:
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(radius), color, thickness)
    return out


def detect_bubbles(mask: NDArray[np.uint8],
                   min_area: float = 0.0,
                   max_area: float = float("inf")) -> dict:
    """Full bubble detection pipeline from a binary mask."""
    mask = _ensure_binary_mask(mask)
    contours, _ = find_contours(mask, mode=cv2.RETR_EXTERNAL)
    contours = filter_contours_by_area(contours, min_area, max_area)
    return {
        "contours": contours,
        "hulls": get_convex_hulls(contours),
        "bounding_rects": get_bounding_rects(contours),
        "bounding_circles": get_bounding_circles(contours),
    }
