"""
Low-level morphological tools for mask processing.
All functions work with binary masks (uint8, values 0 or 255).
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


def _get_kernel(kernel_size: int, kernel_shape: int = cv2.MORPH_ELLIPSE) -> NDArray[np.uint8]:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd positive integer")
    return cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))


def dilate(mask: NDArray, kernel_size: int = 3, iterations: int = 1,
           kernel_shape: int = cv2.MORPH_ELLIPSE) -> NDArray[np.uint8]:
    return cv2.dilate(_ensure_binary_mask(mask), _get_kernel(kernel_size, kernel_shape),
                      iterations=iterations)


def erode(mask: NDArray, kernel_size: int = 3, iterations: int = 1,
          kernel_shape: int = cv2.MORPH_ELLIPSE) -> NDArray[np.uint8]:
    return cv2.erode(_ensure_binary_mask(mask), _get_kernel(kernel_size, kernel_shape),
                     iterations=iterations)


def open_mask(mask: NDArray, kernel_size: int = 3, iterations: int = 1,
              kernel_shape: int = cv2.MORPH_ELLIPSE) -> NDArray[np.uint8]:
    return cv2.morphologyEx(_ensure_binary_mask(mask), cv2.MORPH_OPEN,
                            _get_kernel(kernel_size, kernel_shape), iterations=iterations)


def close_mask(mask: NDArray, kernel_size: int = 3, iterations: int = 1,
               kernel_shape: int = cv2.MORPH_ELLIPSE) -> NDArray[np.uint8]:
    return cv2.morphologyEx(_ensure_binary_mask(mask), cv2.MORPH_CLOSE,
                            _get_kernel(kernel_size, kernel_shape), iterations=iterations)


def clean_noise(mask: NDArray, open_kernel_size: int = 3,
                close_kernel_size: int | None = None, iterations: int = 1,
                kernel_shape: int = cv2.MORPH_ELLIPSE) -> NDArray[np.uint8]:
    """Remove noise: opening (kill small spots) then closing (fill holes)."""
    if close_kernel_size is None:
        close_kernel_size = open_kernel_size
    mask = _ensure_binary_mask(mask)
    mask = open_mask(mask, open_kernel_size, iterations, kernel_shape)
    mask = close_mask(mask, close_kernel_size, iterations, kernel_shape)
    return mask


def improve_mask(mask: NDArray, blur_ksize: int = 3,
                 blur_sigma: float = 0.0, threshold: int = 127) -> NDArray[np.uint8]:
    """Smooth mask edges with Gaussian blur then re-threshold."""
    if blur_ksize % 2 == 0 or blur_ksize < 1:
        raise ValueError("blur_ksize must be an odd positive integer")
    mask = _ensure_binary_mask(mask)
    blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), blur_sigma)
    _, result = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return result


def apply_morphology(mask: NDArray, operation: str, kernel_size: int = 3,
                     iterations: int = 1,
                     kernel_shape: int = cv2.MORPH_ELLIPSE) -> NDArray[np.uint8]:
    """Generic dispatcher: 'dilate' | 'erode' | 'open' | 'close'."""
    ops = {"dilate": dilate, "erode": erode, "open": open_mask, "close": close_mask}
    if operation not in ops:
        raise ValueError(f"operation must be one of {list(ops)}, got '{operation}'")
    return ops[operation](mask, kernel_size=kernel_size, iterations=iterations,
                          kernel_shape=kernel_shape)
