"""
core.py - Core utilities for PatchMatch inpainting
"""

import numpy as np
import math
from typing import Tuple

# ============================================================================
# NUMBA DETECTION
# ============================================================================
def _check_numba():
    try:
        from numba import jit, njit, prange
        return True
    except ImportError:
        return False

HAS_NUMBA = _check_numba()

if HAS_NUMBA:
    from numba import jit, njit, prange
else:
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator if args and callable(args[0]) else decorator
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator if args and callable(args[0]) else decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)


# ============================================================================
# NNF STRUCTURE
# ============================================================================
class NNF:
    __slots__ = ('y', 'x', 'cost', 'rot_idx', 'scale_idx', 'k', 'h', 'w')

    def __init__(self, h: int, w: int, k: int = 5):
        self.h, self.w, self.k = h, w, k
        self.y = np.zeros((h, w, k), dtype=np.float32)
        self.x = np.zeros((h, w, k), dtype=np.float32)
        self.cost = np.full((h, w, k), np.inf, dtype=np.float32)
        self.rot_idx = -np.ones((h, w, k), dtype=np.int8)
        self.scale_idx = -np.ones((h, w, k), dtype=np.int8)

    def reallocate(self, new_h: int, new_w: int):
        self.h, self.w = new_h, new_w
        self.y = np.zeros((new_h, new_w, self.k), dtype=np.float32)
        self.x = np.zeros((new_h, new_w, self.k), dtype=np.float32)
        self.cost = np.full((new_h, new_w, self.k), np.inf, dtype=np.float32)
        self.rot_idx = -np.ones((new_h, new_w, self.k), dtype=np.int8)
        self.scale_idx = -np.ones((new_h, new_w, self.k), dtype=np.int8)


# ============================================================================
# CORE SAMPLING / PATCH EXTRACTION
# ============================================================================
@njit(cache=True)
def sample_pixel(img, sy, sx):
    """Bilinear sample a pixel from image (H,W,C)."""
    h, w, c = img.shape
    if h == 1:
        sy = 0.0
    else:
        sy = min(max(sy, 0.0), h - 1 - 1e-6)
    if w == 1:
        sx = 0.0
    else:
        sx = min(max(sx, 0.0), w - 1 - 1e-6)
    y0, x0 = int(math.floor(sy)), int(math.floor(sx))
    y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
    wy, wx = sy - y0, sx - x0
    pixel = np.zeros(c, dtype=np.float32)
    for ch in range(c):
        pixel[ch] = ((1 - wy) * (1 - wx) * img[y0, x0, ch] +
                     wy * (1 - wx) * img[y1, x0, ch] +
                     (1 - wy) * wx * img[y0, x1, ch] +
                     wy * wx * img[y1, x1, ch])
    return pixel


@njit(cache=True)
def extract_patch_bilinear(img, sy, sx, patch_size):
    """Extract a patch of given size at (sy,sx) using bilinear sampling."""
    c = img.shape[2]
    half = (patch_size - 1) / 2.0
    patch = np.zeros((patch_size, patch_size, c), dtype=np.float32)
    for i in range(patch_size):
        for j in range(patch_size):
            patch[i, j] = sample_pixel(img, sy - half + i, sx - half + j)
    return patch


@njit(cache=True)
def extract_mask_region(mask_pad, sy, sx, patch_size):
    """Extract boolean mask region around (sy,sx) in padded coordinates."""
    half = patch_size // 2
    sy_int = int(round(sy))
    sx_int = int(round(sx))
    h, w = mask_pad.shape
    region = np.ones((patch_size, patch_size), dtype=np.bool_)
    for i in range(patch_size):
        for j in range(patch_size):
            py = sy_int - half + i
            px = sx_int - half + j
            if 0 <= py < h and 0 <= px < w:
                region[i, j] = not mask_pad[py, px]  # True = valid pixel
            else:
                region[i, j] = False
    return region


# ============================================================================
# SSD WITH EARLY EXIT
# ============================================================================
@njit(cache=True)
def patch_ssd(patch1, patch2, target_mask, source_mask, best_cost=1e18):
    """
    Compute normalized SSD between two patches.
    Only pixels valid in BOTH masks are counted.
    Early exit if accumulated cost exceeds best_cost.
    """
    h, w, c = patch1.shape
    ssd = 0.0
    valid_pixels = 0
    max_valid = 0
    # Count total possible valid pixels for normalization
    for i in range(h):
        for j in range(w):
            if target_mask[i, j] and source_mask[i, j]:
                max_valid += 1
    if max_valid == 0:
        return 1e18

    threshold = best_cost * max_valid * c

    for i in range(h):
        for j in range(w):
            if target_mask[i, j] and source_mask[i, j]:
                valid_pixels += 1
                for ch in range(c):
                    diff = patch1[i, j, ch] - patch2[i, j, ch]
                    ssd += diff * diff
                if valid_pixels % 8 == 0 and ssd > threshold:
                    return 1e18
    return ssd / (valid_pixels * c) if valid_pixels > 0 else 1e18


@njit(cache=True)
def patch_ssd_gradient(patch1, patch2, target_mask, source_mask,
                       grad1_x, grad1_y, grad2_x, grad2_y, weight, best_cost=1e18):
    h, w, c = patch1.shape
    total = 0.0
    valid_pixels = 0
    max_valid = 0
    for i in range(h):
        for j in range(w):
            if target_mask[i, j] and source_mask[i, j]:
                max_valid += 1
    if max_valid == 0:
        return 1e18

    threshold = best_cost * max_valid * c * (1 + 2 * weight)

    for i in range(h):
        for j in range(w):
            if target_mask[i, j] and source_mask[i, j]:
                valid_pixels += 1
                for ch in range(c):
                    diff = patch1[i, j, ch] - patch2[i, j, ch]
                    total += diff * diff
                for ch in range(c):
                    diff_x = grad1_x[i, j, ch] - grad2_x[i, j, ch]
                    diff_y = grad1_y[i, j, ch] - grad2_y[i, j, ch]
                    total += weight * (diff_x * diff_x + diff_y * diff_y)
                if valid_pixels % 8 == 0 and total > threshold:
                    return 1e18
    return total / (valid_pixels * c) if valid_pixels > 0 else 1e18


# ============================================================================
# GRADIENTS
# ============================================================================
@njit(cache=True)
def compute_gradients(img):
    h, w, c = img.shape
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            for ch in range(c):
                grad_x[i, j, ch] = (img[i, j + 1, ch] - img[i, j - 1, ch]) * 0.5
                grad_y[i, j, ch] = (img[i + 1, j, ch] - img[i - 1, j, ch]) * 0.5
    return grad_x, grad_y


# ============================================================================
# TRANSFORMS (rotation, scaling, resize)
# ============================================================================
@njit(cache=True)
def rotate_patch_fast(patch, rot):
    h, w, c = patch.shape
    if rot == 0:
        return patch.copy()
    elif rot == 90:
        out = np.zeros((w, h, c), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                out[j, h - 1 - i] = patch[i, j]
        return out
    elif rot == 180:
        out = np.zeros((h, w, c), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                out[h - 1 - i, w - 1 - j] = patch[i, j]
        return out
    elif rot == 270:
        out = np.zeros((w, h, c), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                out[w - 1 - j, i] = patch[i, j]
        return out
    else:
        return patch.copy()


@njit(cache=True)
def transform_patch_generic(patch, rotation, scale):
    h, w, c = patch.shape
    if abs(rotation) < 1e-6 and abs(scale - 1.0) < 1e-6:
        return patch.copy()
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    src_cx, src_cy = (w - 1) / 2.0, (h - 1) / 2.0
    dst_cx, dst_cy = (new_w - 1) / 2.0, (new_h - 1) / 2.0
    out = np.zeros((new_h, new_w, c), dtype=np.float32)
    rad = rotation * math.pi / 180.0
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    for i in range(new_h):
        for j in range(new_w):
            rel_x = j - dst_cx
            rel_y = i - dst_cy
            if abs(scale - 1.0) > 1e-6:
                rel_x /= scale
                rel_y /= scale
            src_rel_x = cos_a * rel_x - sin_a * rel_y
            src_rel_y = sin_a * rel_x + cos_a * rel_y
            src_x = src_cx + src_rel_x
            src_y = src_cy + src_rel_y
            x0 = int(math.floor(src_x))
            y0 = int(math.floor(src_y))
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            wx = src_x - x0
            wy = src_y - y0
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
                out[i, j] = ((1 - wy) * (1 - wx) * patch[y0, x0] +
                             wy * (1 - wx) * patch[y1, x0] +
                             (1 - wy) * wx * patch[y0, x1] +
                             wy * wx * patch[y1, x1])
    return out


@njit(cache=True)
def resize_patch_to(patch, target_h, target_w):
    src_h, src_w, c = patch.shape
    if src_h == target_h and src_w == target_w:
        return patch.copy()
    out = np.zeros((target_h, target_w, c), dtype=np.float32)
    scale_h, scale_w = src_h / target_h, src_w / target_w
    for i in range(target_h):
        for j in range(target_w):
            src_y = (i + 0.5) * scale_h - 0.5
            src_x = (j + 0.5) * scale_w - 0.5
            y0 = int(math.floor(src_y))
            x0 = int(math.floor(src_x))
            y1 = min(y0 + 1, src_h - 1)
            x1 = min(x0 + 1, src_w - 1)
            wy = src_y - y0
            wx = src_x - x0
            if 0 <= y0 < src_h and 0 <= x0 < src_w:
                out[i, j] = ((1 - wy) * (1 - wx) * patch[y0, x0] +
                             wy * (1 - wx) * patch[y1, x0] +
                             (1 - wy) * wx * patch[y0, x1] +
                             wy * wx * patch[y1, x1])
    return out


# ============================================================================
# PRECOMPUTED TRANSFORM MAPS
# ============================================================================
@njit(cache=True)
def precompute_transform_maps(patch_size, rotations, scales):
    n_rots = len(rotations)
    n_scales = len(scales)
    N = n_rots * n_scales
    half = (patch_size - 1) / 2.0
    maps_x = np.zeros((N, patch_size, patch_size), dtype=np.float32)
    maps_y = np.zeros((N, patch_size, patch_size), dtype=np.float32)
    w00 = np.zeros((N, patch_size, patch_size), dtype=np.float32)
    w01 = np.zeros((N, patch_size, patch_size), dtype=np.float32)
    w10 = np.zeros((N, patch_size, patch_size), dtype=np.float32)
    w11 = np.zeros((N, patch_size, patch_size), dtype=np.float32)
    idx = 0
    for ri in range(n_rots):
        rot = rotations[ri]
        rad = rot * math.pi / 180.0
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        for si in range(n_scales):
            scale = scales[si]
            for i in range(patch_size):
                for j in range(patch_size):
                    rel_x = j - half
                    rel_y = i - half
                    if abs(scale - 1.0) > 1e-6:
                        rel_x /= scale
                        rel_y /= scale
                    src_x = half + cos_a * rel_x - sin_a * rel_y
                    src_y = half + sin_a * rel_x + cos_a * rel_y
                    x0 = int(math.floor(src_x))
                    y0 = int(math.floor(src_y))
                    x1 = min(x0 + 1, patch_size - 1)
                    y1 = min(y0 + 1, patch_size - 1)
                    x0 = max(0, min(x0, patch_size - 1))
                    y0 = max(0, min(y0, patch_size - 1))
                    wx = src_x - x0
                    wy = src_y - y0
                    maps_x[idx, i, j] = x0
                    maps_y[idx, i, j] = y0
                    w00[idx, i, j] = (1 - wy) * (1 - wx)
                    w01[idx, i, j] = wy * (1 - wx)
                    w10[idx, i, j] = (1 - wy) * wx
                    w11[idx, i, j] = wy * wx
            idx += 1
    return maps_x, maps_y, w00, w01, w10, w11


@njit(cache=True)
def apply_cached_transform(patch, map_x, map_y, w00, w01, w10, w11):
    h, w, c = patch.shape
    out = np.zeros((h, w, c), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            x0 = int(map_x[i, j])
            y0 = int(map_y[i, j])
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            for ch in range(c):
                out[i, j, ch] = (w00[i, j] * patch[y0, x0, ch] +
                                 w01[i, j] * patch[y1, x0, ch] +
                                 w10[i, j] * patch[y0, x1, ch] +
                                 w11[i, j] * patch[y1, x1, ch])
    return out


# ============================================================================
# KNN HELPERS
# ============================================================================
@njit(cache=True)
def is_duplicate(y_arr, x_arr, rot_idx_arr, scale_idx_arr, y, x, k, sy_off, sx_off, ri, si):
    for i in range(k):
        if (abs(y_arr[y, x, i] - sy_off) < 1e-4 and abs(x_arr[y, x, i] - sx_off) < 1e-4 and
                rot_idx_arr[y, x, i] == ri and scale_idx_arr[y, x, i] == si):
            return True
    return False


@njit(cache=True)
def sort_knn_row(y_arr, x_arr, cost_arr, rot_idx_arr, scale_idx_arr, y, x, k):
    for i in range(1, k):
        for j in range(i, 0, -1):
            if cost_arr[y, x, j] < cost_arr[y, x, j - 1]:
                for arr in (y_arr, x_arr, cost_arr, rot_idx_arr, scale_idx_arr):
                    arr[y, x, j], arr[y, x, j - 1] = arr[y, x, j - 1], arr[y, x, j]
            else:
                break


@njit(cache=True)
def update_knn(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx,
               y, x, sy_off, sx_off, cost, k, ri, si):
    if is_duplicate(nnf_y, nnf_x, rot_idx, scale_idx, y, x, k, sy_off, sx_off, ri, si):
        return False
    worst_idx, worst_cost = 0, nnf_cost[y, x, 0]
    for i in range(1, k):
        if nnf_cost[y, x, i] > worst_cost:
            worst_cost = nnf_cost[y, x, i]
            worst_idx = i
    if cost < worst_cost:
        nnf_y[y, x, worst_idx] = sy_off
        nnf_x[y, x, worst_idx] = sx_off
        nnf_cost[y, x, worst_idx] = cost
        rot_idx[y, x, worst_idx] = ri
        scale_idx[y, x, worst_idx] = si
        sort_knn_row(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx, y, x, k)
        return True
    return False


@njit(cache=True)
def find_rot_scale_index(rot, scale, rotations, scales):
    ri, si = -1, -1
    for i in range(len(rotations)):
        if abs(rotations[i] - rot) < 1e-6:
            ri = i
            break
    for j in range(len(scales)):
        if abs(scales[j] - scale) < 1e-6:
            si = j
            break
    return ri, si