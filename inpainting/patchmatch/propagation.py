"""
propagation.py - PatchMatch propagation, random search, coherence, and initialization
"""

import numpy as np
import math
from numba import njit, prange

from .core import (
    extract_patch_bilinear, extract_mask_region,
    patch_ssd, patch_ssd_gradient,
    rotate_patch_fast, transform_patch_generic, resize_patch_to,
    apply_cached_transform,
    update_knn, find_rot_scale_index, sort_knn_row
)


# ============================================================================
# INITIALIZATION
# ============================================================================
@njit(cache=True, parallel=False)
def initialize_nnf(img_pad, mask_pad, h, w, patch_size, k,
                   rotations, scales, valid_sources):
    pad = patch_size // 2
    nnf_y = np.zeros((h, w, k), dtype=np.float32)
    nnf_x = np.zeros((h, w, k), dtype=np.float32)
    nnf_cost = np.full((h, w, k), 1e18, dtype=np.float32)
    rot_idx = -np.ones((h, w, k), dtype=np.int8)
    scale_idx = -np.ones((h, w, k), dtype=np.int8)

    n_sources = valid_sources.shape[0]
    if n_sources == 0:
        return nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx

    max_attempts = max(200, k * 30)
    for y in range(h):
        for x in range(w):
            if mask_pad[y + pad, x + pad]:  # hole
                found = 0
                attempts = 0
                while found < k and attempts < max_attempts:
                    idx = np.random.randint(0, n_sources)
                    sy = float(valid_sources[idx, 0])
                    sx = float(valid_sources[idx, 1])
                    sy_int, sx_int = int(sy), int(sx)
                    if (0 <= sy_int < h and 0 <= sx_int < w and
                            not mask_pad[sy_int + pad, sx_int + pad]):
                        cost, best_rot, best_scale = _compute_patch_distance(
                            img_pad, mask_pad, y, x, sy, sx, patch_size, rotations, scales)
                        ri, si = find_rot_scale_index(best_rot, best_scale, rotations, scales)
                        if ri >= 0 and si >= 0:
                            if update_knn(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx,
                                          y, x, sy - float(y), sx - float(x), cost, k, ri, si):
                                found += 1
                    attempts += 1
    return nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx


@njit(cache=True)
def _compute_patch_distance(img_pad, mask_pad, y, x, sy, sx, patch_size, rotations, scales):
    pad = patch_size // 2
    target = img_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
    target_mask = ~mask_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
    src = extract_patch_bilinear(img_pad, sy + pad, sx + pad, patch_size)
    src_mask = extract_mask_region(mask_pad, sy + pad, sx + pad, patch_size)
    best_cost = 1e18
    best_rot, best_scale = 0.0, 1.0
    for rot in rotations:
        for scale in scales:
            if rot in (0, 90, 180, 270) and abs(scale - 1.0) < 1e-6:
                transformed = rotate_patch_fast(src, rot)
            else:
                transformed = transform_patch_generic(src, rot, scale)
            if transformed.shape[0] != patch_size or transformed.shape[1] != patch_size:
                transformed = resize_patch_to(transformed, patch_size, patch_size)
            cost = patch_ssd(target, transformed, target_mask, src_mask, best_cost)
            if cost < best_cost:
                best_cost, best_rot, best_scale = cost, rot, scale
    return best_cost, best_rot, best_scale


# ============================================================================
# COST RECOMPUTATION (parallel)
# ============================================================================
@njit(cache=True, parallel=True)
def recompute_nnf_costs(img_pad, mask_pad, nnf_y, nnf_x, nnf_cost,
                        rot_idx, scale_idx, h, w, patch_size, k,
                        rotations, scales, transform_maps,
                        use_features, feature_weight,
                        grad_x_full, grad_y_full):
    pad = patch_size // 2
    maps_x, maps_y, w00, w01, w10, w11 = transform_maps
    n_rots, n_scales = len(rotations), len(scales)
    for y in prange(h):
        for x in range(w):
            if not mask_pad[y + pad, x + pad]:
                continue
            target = img_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_mask = ~mask_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_grad_x, target_grad_y = None, None
            if use_features:
                target_grad_x = grad_x_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
                target_grad_y = grad_y_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            best_among_k = nnf_cost[y, x, 0]
            for ki in range(k):
                sy = float(y) + nnf_y[y, x, ki]
                sx = float(x) + nnf_x[y, x, ki]
                if 0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1):
                    sy_int, sx_int = int(round(sy)), int(round(sx))
                    if (0 <= sy_int < h and 0 <= sx_int < w and
                            not mask_pad[sy_int + pad, sx_int + pad]):
                        ri, si = rot_idx[y, x, ki], scale_idx[y, x, ki]
                        if ri >= 0 and si >= 0:
                            tidx = ri * n_scales + si
                            src = extract_patch_bilinear(img_pad, sy + pad, sx + pad, patch_size)
                            transformed = apply_cached_transform(src, maps_x[tidx], maps_y[tidx],
                                                                 w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            src_mask = extract_mask_region(mask_pad, sy + pad, sx + pad, patch_size)
                            if use_features:
                                src_grad_x = grad_x_full[sy_int + pad:sy_int + pad + patch_size,
                                             sx_int + pad:sx_int + pad + patch_size]
                                src_grad_y = grad_y_full[sy_int + pad:sy_int + pad + patch_size,
                                             sx_int + pad:sx_int + pad + patch_size]
                                transformed_grad_x = apply_cached_transform(src_grad_x, maps_x[tidx], maps_y[tidx],
                                                                             w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                                transformed_grad_y = apply_cached_transform(src_grad_y, maps_x[tidx], maps_y[tidx],
                                                                             w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                                cost = patch_ssd_gradient(target, transformed, target_mask, src_mask,
                                                          target_grad_x, target_grad_y,
                                                          transformed_grad_x, transformed_grad_y,
                                                          feature_weight, best_among_k)
                            else:
                                cost = patch_ssd(target, transformed, target_mask, src_mask, best_among_k)
                            nnf_cost[y, x, ki] = cost
                            if cost < best_among_k:
                                best_among_k = cost
                        else:
                            nnf_cost[y, x, ki] = 1e18
                    else:
                        nnf_cost[y, x, ki] = 1e18
                else:
                    nnf_cost[y, x, ki] = 1e18


# ============================================================================
# PROPAGATION FORWARD
# ============================================================================
@njit(cache=True, parallel=False)
def propagate_forward(img_pad, mask_pad, nnf_y, nnf_x, nnf_cost,
                      rot_idx, scale_idx, h, w, patch_size, k,
                      rotations, scales, transform_maps,
                      use_features, feature_weight,
                      grad_x_full, grad_y_full):
    pad = patch_size // 2
    maps_x, maps_y, w00, w01, w10, w11 = transform_maps
    n_rots, n_scales = len(rotations), len(scales)

    for y in range(h):
        for x in range(w):
            if not mask_pad[y + pad, x + pad]:
                continue
            target = img_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_mask = ~mask_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_grad_x, target_grad_y = None, None
            if use_features:
                target_grad_x = grad_x_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
                target_grad_y = grad_y_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]

            # Left neighbor
            if x > 0:
                for ki in range(k):
                    sy_off, sx_off = nnf_y[y, x - 1, ki], nnf_x[y, x - 1, ki]
                    ri, si = rot_idx[y, x - 1, ki], scale_idx[y, x - 1, ki]
                    if ri < 0 or si < 0:
                        continue
                    rot, scale = rotations[ri], scales[si]
                    rad = rot * math.pi / 180.0
                    step_x = math.cos(rad) * scale
                    step_y = math.sin(rad) * scale
                    sy = float(y) + sy_off + step_y
                    sx = float(x - 1) + sx_off + step_x
                    sy_int, sx_int = int(round(sy)), int(round(sx))
                    if (0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1) and
                            0 <= sy_int < h and 0 <= sx_int < w and
                            not mask_pad[sy_int + pad, sx_int + pad]):
                        tidx = ri * n_scales + si
                        src = extract_patch_bilinear(img_pad, sy + pad, sx + pad, patch_size)
                        transformed = apply_cached_transform(src, maps_x[tidx], maps_y[tidx],
                                                              w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                        src_mask = extract_mask_region(mask_pad, sy + pad, sx + pad, patch_size)
                        best_among_k = nnf_cost[y, x, k - 1]
                        if use_features:
                            src_grad_x = grad_x_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            src_grad_y = grad_y_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            transformed_grad_x = apply_cached_transform(src_grad_x, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            transformed_grad_y = apply_cached_transform(src_grad_y, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            cost = patch_ssd_gradient(target, transformed, target_mask, src_mask,
                                                       target_grad_x, target_grad_y,
                                                       transformed_grad_x, transformed_grad_y,
                                                       feature_weight, best_among_k)
                        else:
                            cost = patch_ssd(target, transformed, target_mask, src_mask, best_among_k)
                        if cost < best_among_k:
                            update_knn(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx,
                                       y, x, sy - float(y), sx - float(x), cost, k, ri, si)

            # Top neighbor
            if y > 0:
                for ki in range(k):
                    sy_off, sx_off = nnf_y[y - 1, x, ki], nnf_x[y - 1, x, ki]
                    ri, si = rot_idx[y - 1, x, ki], scale_idx[y - 1, x, ki]
                    if ri < 0 or si < 0:
                        continue
                    rot, scale = rotations[ri], scales[si]
                    rad = rot * math.pi / 180.0
                    step_x = -math.sin(rad) * scale
                    step_y = math.cos(rad) * scale
                    sy = float(y - 1) + sy_off + step_y
                    sx = float(x) + sx_off + step_x
                    sy_int, sx_int = int(round(sy)), int(round(sx))
                    if (0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1) and
                            0 <= sy_int < h and 0 <= sx_int < w and
                            not mask_pad[sy_int + pad, sx_int + pad]):
                        tidx = ri * n_scales + si
                        src = extract_patch_bilinear(img_pad, sy + pad, sx + pad, patch_size)
                        transformed = apply_cached_transform(src, maps_x[tidx], maps_y[tidx],
                                                              w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                        src_mask = extract_mask_region(mask_pad, sy + pad, sx + pad, patch_size)
                        best_among_k = nnf_cost[y, x, k - 1]
                        if use_features:
                            src_grad_x = grad_x_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            src_grad_y = grad_y_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            transformed_grad_x = apply_cached_transform(src_grad_x, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            transformed_grad_y = apply_cached_transform(src_grad_y, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            cost = patch_ssd_gradient(target, transformed, target_mask, src_mask,
                                                       target_grad_x, target_grad_y,
                                                       transformed_grad_x, transformed_grad_y,
                                                       feature_weight, best_among_k)
                        else:
                            cost = patch_ssd(target, transformed, target_mask, src_mask, best_among_k)
                        if cost < best_among_k:
                            update_knn(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx,
                                       y, x, sy - float(y), sx - float(x), cost, k, ri, si)


# ============================================================================
# PROPAGATION BACKWARD
# ============================================================================
@njit(cache=True, parallel=False)
def propagate_backward(img_pad, mask_pad, nnf_y, nnf_x, nnf_cost,
                       rot_idx, scale_idx, h, w, patch_size, k,
                       rotations, scales, transform_maps,
                       use_features, feature_weight,
                       grad_x_full, grad_y_full):
    pad = patch_size // 2
    maps_x, maps_y, w00, w01, w10, w11 = transform_maps
    n_rots, n_scales = len(rotations), len(scales)

    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if not mask_pad[y + pad, x + pad]:
                continue
            target = img_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_mask = ~mask_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_grad_x, target_grad_y = None, None
            if use_features:
                target_grad_x = grad_x_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
                target_grad_y = grad_y_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]

            # Right neighbor
            if x < w - 1:
                for ki in range(k):
                    sy_off, sx_off = nnf_y[y, x + 1, ki], nnf_x[y, x + 1, ki]
                    ri, si = rot_idx[y, x + 1, ki], scale_idx[y, x + 1, ki]
                    if ri < 0 or si < 0:
                        continue
                    rot, scale = rotations[ri], scales[si]
                    rad = rot * math.pi / 180.0
                    step_x = -math.cos(rad) * scale
                    step_y = -math.sin(rad) * scale
                    sy = float(y) + sy_off + step_y
                    sx = float(x + 1) + sx_off + step_x
                    sy_int, sx_int = int(round(sy)), int(round(sx))
                    if (0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1) and
                            0 <= sy_int < h and 0 <= sx_int < w and
                            not mask_pad[sy_int + pad, sx_int + pad]):
                        tidx = ri * n_scales + si
                        src = extract_patch_bilinear(img_pad, sy + pad, sx + pad, patch_size)
                        transformed = apply_cached_transform(src, maps_x[tidx], maps_y[tidx],
                                                              w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                        src_mask = extract_mask_region(mask_pad, sy + pad, sx + pad, patch_size)
                        best_among_k = nnf_cost[y, x, k - 1]
                        if use_features:
                            src_grad_x = grad_x_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            src_grad_y = grad_y_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            transformed_grad_x = apply_cached_transform(src_grad_x, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            transformed_grad_y = apply_cached_transform(src_grad_y, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            cost = patch_ssd_gradient(target, transformed, target_mask, src_mask,
                                                       target_grad_x, target_grad_y,
                                                       transformed_grad_x, transformed_grad_y,
                                                       feature_weight, best_among_k)
                        else:
                            cost = patch_ssd(target, transformed, target_mask, src_mask, best_among_k)
                        if cost < best_among_k:
                            update_knn(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx,
                                       y, x, sy - float(y), sx - float(x), cost, k, ri, si)

            # Bottom neighbor
            if y < h - 1:
                for ki in range(k):
                    sy_off, sx_off = nnf_y[y + 1, x, ki], nnf_x[y + 1, x, ki]
                    ri, si = rot_idx[y + 1, x, ki], scale_idx[y + 1, x, ki]
                    if ri < 0 or si < 0:
                        continue
                    rot, scale = rotations[ri], scales[si]
                    rad = rot * math.pi / 180.0
                    step_x = math.sin(rad) * scale
                    step_y = -math.cos(rad) * scale
                    sy = float(y + 1) + sy_off + step_y
                    sx = float(x) + sx_off + step_x
                    sy_int, sx_int = int(round(sy)), int(round(sx))
                    if (0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1) and
                            0 <= sy_int < h and 0 <= sx_int < w and
                            not mask_pad[sy_int + pad, sx_int + pad]):
                        tidx = ri * n_scales + si
                        src = extract_patch_bilinear(img_pad, sy + pad, sx + pad, patch_size)
                        transformed = apply_cached_transform(src, maps_x[tidx], maps_y[tidx],
                                                              w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                        src_mask = extract_mask_region(mask_pad, sy + pad, sx + pad, patch_size)
                        best_among_k = nnf_cost[y, x, k - 1]
                        if use_features:
                            src_grad_x = grad_x_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            src_grad_y = grad_y_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            transformed_grad_x = apply_cached_transform(src_grad_x, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            transformed_grad_y = apply_cached_transform(src_grad_y, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            cost = patch_ssd_gradient(target, transformed, target_mask, src_mask,
                                                       target_grad_x, target_grad_y,
                                                       transformed_grad_x, transformed_grad_y,
                                                       feature_weight, best_among_k)
                        else:
                            cost = patch_ssd(target, transformed, target_mask, src_mask, best_among_k)
                        if cost < best_among_k:
                            update_knn(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx,
                                       y, x, sy - float(y), sx - float(x), cost, k, ri, si)


# ============================================================================
# RANDOM SEARCH
# ============================================================================
@njit(cache=True, parallel=False)
def random_search(img_pad, mask_pad, nnf_y, nnf_x, nnf_cost,
                  rot_idx, scale_idx, h, w, patch_size, k, rotations, scales,
                  transform_maps, search_radius, random_decay,
                  use_features, feature_weight,
                  grad_x_full, grad_y_full):
    pad = patch_size // 2
    maps_x, maps_y, w00, w01, w10, w11 = transform_maps
    n_rots, n_scales = len(rotations), len(scales)

    for y in range(h):
        for x in range(w):
            if not mask_pad[y + pad, x + pad]:
                continue
            target = img_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_mask = ~mask_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_grad_x, target_grad_y = None, None
            if use_features:
                target_grad_x = grad_x_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
                target_grad_y = grad_y_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]

            radius = search_radius
            while radius > 1:
                dy = np.random.randint(-radius, radius + 1)
                dx = np.random.randint(-radius, radius + 1)
                sy = float(y) + nnf_y[y, x, 0] + float(dy)
                sx = float(x) + nnf_x[y, x, 0] + float(dx)
                sy_int, sx_int = int(round(sy)), int(round(sx))
                if (0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1) and
                        0 <= sy_int < h and 0 <= sx_int < w and
                        not mask_pad[sy_int + pad, sx_int + pad]):
                    ri = rot_idx[y, x, 0]
                    si = scale_idx[y, x, 0]
                    rot = rotations[ri]
                    scale = scales[si]
                    if np.random.random() < 0.05:
                        ri = np.random.randint(0, n_rots)
                        rot = rotations[ri]
                    if np.random.random() < 0.05:
                        si = np.random.randint(0, n_scales)
                        scale = scales[si]
                    if ri >= 0 and si >= 0:
                        tidx = ri * n_scales + si
                        src = extract_patch_bilinear(img_pad, sy + pad, sx + pad, patch_size)
                        transformed = apply_cached_transform(src, maps_x[tidx], maps_y[tidx],
                                                              w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                        src_mask = extract_mask_region(mask_pad, sy + pad, sx + pad, patch_size)
                        best_among_k = nnf_cost[y, x, k - 1]
                        if use_features:
                            src_grad_x = grad_x_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            src_grad_y = grad_y_full[sy_int + pad:sy_int + pad + patch_size,
                                         sx_int + pad:sx_int + pad + patch_size]
                            transformed_grad_x = apply_cached_transform(src_grad_x, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            transformed_grad_y = apply_cached_transform(src_grad_y, maps_x[tidx], maps_y[tidx],
                                                                         w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            cost = patch_ssd_gradient(target, transformed, target_mask, src_mask,
                                                       target_grad_x, target_grad_y,
                                                       transformed_grad_x, transformed_grad_y,
                                                       feature_weight, best_among_k)
                        else:
                            cost = patch_ssd(target, transformed, target_mask, src_mask, best_among_k)
                        if cost < best_among_k:
                            update_knn(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx,
                                       y, x, sy - float(y), sx - float(x), cost, k, ri, si)
                radius = int(radius * random_decay)


# ============================================================================
# COHERENCE SEARCH
# ============================================================================
@njit(cache=True, parallel=False)
def coherence_search(img_pad, mask_pad, nnf_y, nnf_x, nnf_cost,
                     rot_idx, scale_idx, h, w, patch_size, k, rotations, scales,
                     transform_maps, use_features, feature_weight,
                     grad_x_full, grad_y_full):
    pad = patch_size // 2
    maps_x, maps_y, w00, w01, w10, w11 = transform_maps
    n_rots, n_scales = len(rotations), len(scales)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not mask_pad[y + pad, x + pad]:
                continue
            target = img_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_mask = ~mask_pad[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
            target_grad_x, target_grad_y = None, None
            if use_features:
                target_grad_x = grad_x_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]
                target_grad_y = grad_y_full[y + pad:y + pad + patch_size, x + pad:x + pad + patch_size]

            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    for ki in range(k):
                        sy_off, sx_off = nnf_y[ny, nx, ki], nnf_x[ny, nx, ki]
                        ri, si = rot_idx[ny, nx, ki], scale_idx[ny, nx, ki]
                        if ri < 0 or si < 0:
                            continue
                        src_y_abs = float(ny) + sy_off
                        src_x_abs = float(nx) + sx_off
                        sy = src_y_abs + (y - ny)
                        sx = src_x_abs + (x - nx)
                        sy_int, sx_int = int(round(sy)), int(round(sx))
                        if (0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1) and
                                0 <= sy_int < h and 0 <= sx_int < w and
                                not mask_pad[sy_int + pad, sx_int + pad]):
                            tidx = ri * n_scales + si
                            src = extract_patch_bilinear(img_pad, sy + pad, sx + pad, patch_size)
                            transformed = apply_cached_transform(src, maps_x[tidx], maps_y[tidx],
                                                                  w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                            src_mask = extract_mask_region(mask_pad, sy + pad, sx + pad, patch_size)
                            best_among_k = nnf_cost[y, x, k - 1]
                            if use_features:
                                src_grad_x = grad_x_full[sy_int + pad:sy_int + pad + patch_size,
                                             sx_int + pad:sx_int + pad + patch_size]
                                src_grad_y = grad_y_full[sy_int + pad:sy_int + pad + patch_size,
                                             sx_int + pad:sx_int + pad + patch_size]
                                transformed_grad_x = apply_cached_transform(src_grad_x, maps_x[tidx], maps_y[tidx],
                                                                             w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                                transformed_grad_y = apply_cached_transform(src_grad_y, maps_x[tidx], maps_y[tidx],
                                                                             w00[tidx], w01[tidx], w10[tidx], w11[tidx])
                                cost = patch_ssd_gradient(target, transformed, target_mask, src_mask,
                                                           target_grad_x, target_grad_y,
                                                           transformed_grad_x, transformed_grad_y,
                                                           feature_weight, best_among_k)
                            else:
                                cost = patch_ssd(target, transformed, target_mask, src_mask, best_among_k)
                            if cost < best_among_k:
                                update_knn(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx,
                                           y, x, sy - float(y), sx - float(x), cost, k, ri, si)


# ============================================================================
# BIDIRECTIONAL HEURISTIC
# ============================================================================
@njit(cache=True, parallel=False)
def bidirectional_heuristic(nnf_y, nnf_x, nnf_cost,
                            rot_idx, scale_idx, h, w, k):
    bwd_y = np.zeros((h, w), dtype=np.float32)
    bwd_x = np.zeros((h, w), dtype=np.float32)
    bwd_cost = np.full((h, w), 1e18, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            if nnf_cost[y, x, 0] >= 1e17:
                continue
            sy = float(y) + nnf_y[y, x, 0]
            sx = float(x) + nnf_x[y, x, 0]
            if 0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1):
                syi, sxi = int(round(sy)), int(round(sx))
                if 0 <= syi < h and 0 <= sxi < w:
                    if nnf_cost[y, x, 0] < bwd_cost[syi, sxi]:
                        bwd_y[syi, sxi] = -nnf_y[y, x, 0]
                        bwd_x[syi, sxi] = -nnf_x[y, x, 0]
                        bwd_cost[syi, sxi] = nnf_cost[y, x, 0]

    for y in range(h):
        for x in range(w):
            fwd_cost = nnf_cost[y, x, 0]
            bwd_c = bwd_cost[y, x]
            if fwd_cost < 1e17 and bwd_c < fwd_cost * 0.8:
                nnf_y[y, x, 0] = bwd_y[y, x]
                nnf_x[y, x, 0] = bwd_x[y, x]
                nnf_cost[y, x, 0] = bwd_c
                sort_knn_row(nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx, y, x, k)