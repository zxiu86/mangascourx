"""
engine.py - Main PatchMatch inpainting engine
"""

import numpy as np
import math
import time
import warnings
from .core import NNF, precompute_transform_maps, compute_gradients
from .propagation import (
    initialize_nnf, recompute_nnf_costs,
    propagate_forward, propagate_backward,
    random_search, coherence_search, bidirectional_heuristic
)


class PatchMatchInpainter:
    def __init__(self, patch_size=7, pyramid_levels=5, iterations=6, knn=3,
                 random_decay=0.5, use_rotation=True, use_scale=True,
                 use_bidirectional=True, use_coherence=True,
                 use_gaussian_pyramid=True, min_pyramid_size=32,
                 use_features=False, feature_weight=0.5, seed=None,
                 verbose=False):
        self.patch_size = patch_size
        self.pyramid_levels = pyramid_levels
        self.iterations = iterations
        self.knn = knn
        self.random_decay = random_decay
        self.use_rotation = use_rotation
        self.use_scale = use_scale
        self.use_bidirectional = use_bidirectional
        self.use_coherence = use_coherence
        self.use_gaussian_pyramid = use_gaussian_pyramid
        self.min_pyramid_size = min_pyramid_size
        self.use_features = use_features
        self.feature_weight = feature_weight
        self.seed = seed
        self.verbose = verbose

        self.rotations = np.array([0, 90, 180, 270], dtype=np.float32) if use_rotation else np.array([0],
                                                                                                      dtype=np.float32)
        self.scales = np.array([0.8, 1.0, 1.2], dtype=np.float32) if use_scale else np.array([1.0], dtype=np.float32)
        self.transform_maps = None
        self.spatial_kernel = None

    def _build_pyramid(self, image, mask):
        pyramid, mask_pyramid = [], []
        current_img = image.astype(np.float32)
        current_mask = mask.astype(np.bool_)
        pyramid.append(current_img)
        mask_pyramid.append(current_mask)
        try:
            from scipy.ndimage import gaussian_filter
            has_scipy = True
        except ImportError:
            has_scipy = False
        for level in range(1, self.pyramid_levels):
            h, w = current_img.shape[:2]
            new_h = max(self.min_pyramid_size, h // 2)
            new_w = max(self.min_pyramid_size, w // 2)
            if h <= new_h or w <= new_w:
                break
            if has_scipy:
                sigma = 0.5 * (h / new_h)
                blurred = gaussian_filter(current_img, sigma=sigma, truncate=2.0, axes=(0, 1))
            else:
                blurred = current_img
            factor_h, factor_w = h / new_h, w / new_w
            down_img = np.zeros((new_h, new_w, current_img.shape[2]), dtype=np.float32)
            down_mask = np.zeros((new_h, new_w), dtype=np.bool_)
            for y in range(new_h):
                for x in range(new_w):
                    src_y = y * factor_h
                    src_x = x * factor_w
                    y0 = int(math.floor(src_y))
                    x0 = int(math.floor(src_x))
                    y1 = min(y0 + 1, h - 1)
                    x1 = min(x0 + 1, w - 1)
                    wy = src_y - y0
                    wx = src_x - x0
                    down_img[y, x] = ((1 - wy) * (1 - wx) * blurred[y0, x0] +
                                      wy * (1 - wx) * blurred[y1, x0] +
                                      (1 - wy) * wx * blurred[y0, x1] +
                                      wy * wx * blurred[y1, x1])
                    down_mask[y, x] = np.any(current_mask[y0:y1 + 1, x0:x1 + 1])
            current_img, current_mask = down_img, down_mask
            pyramid.append(current_img)
            mask_pyramid.append(current_mask)
        return pyramid, mask_pyramid

    def _upsample_nnf(self, nnf, new_h, new_w):
        old_h, old_w = nnf.h, nnf.w
        scale_h, scale_w = new_h / old_h, new_w / old_w
        new_nnf = NNF(new_h, new_w, nnf.k)
        for y in range(new_h):
            for x in range(new_w):
                src_y = min(int(y / scale_h), old_h - 1)
                src_x = min(int(x / scale_w), old_w - 1)
                for ki in range(nnf.k):
                    new_nnf.y[y, x, ki] = nnf.y[src_y, src_x, ki] * scale_h
                    new_nnf.x[y, x, ki] = nnf.x[src_y, src_x, ki] * scale_w
                    new_nnf.cost[y, x, ki] = nnf.cost[src_y, src_x, ki]
                    new_nnf.rot_idx[y, x, ki] = nnf.rot_idx[src_y, src_x, ki]
                    new_nnf.scale_idx[y, x, ki] = nnf.scale_idx[src_y, src_x, ki]
        return new_nnf

    def _precompute_spatial_kernel(self):
        ps = self.patch_size
        half = ps // 2
        kernel = np.zeros((ps, ps), dtype=np.float32)
        for i in range(ps):
            for j in range(ps):
                dist = math.sqrt((i - half) ** 2 + (j - half) ** 2)
                kernel[i, j] = math.exp(-dist / (ps / 2))
        return kernel

    def run(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        start_time = time.time()
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.verbose:
            print("🧩 PatchMatch Inpainter v10.1-PROD (Complete & Verified)")
            print(f"   Image: {image.shape}, Holes: {np.sum(mask)} px")
            print(f"   Patch: {self.patch_size}, K: {self.knn}, Seed: {self.seed}")
            print(f"   Features: {'✅' if self.use_features else '❌'}")

        mask = mask.astype(np.bool_)
        self.transform_maps = precompute_transform_maps(self.patch_size, self.rotations, self.scales)
        self.spatial_kernel = self._precompute_spatial_kernel()

        pyramid, mask_pyramid = self._build_pyramid(image, mask)
        num_levels = len(pyramid)

        coarse_img = pyramid[-1]
        coarse_mask = mask_pyramid[-1]
        h_c, w_c = coarse_img.shape[:2]
        nnf = NNF(h_c, w_c, k=self.knn)
        pad = self.patch_size // 2

        valid_y, valid_x = np.where(~coarse_mask)
        valid_sources = np.stack((valid_y, valid_x), axis=1).astype(np.int32)

        img_pad = np.pad(coarse_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        mask_pad = np.pad(coarse_mask, ((pad, pad), (pad, pad)), mode='constant', constant_values=False)

        grad_x_full, grad_y_full = None, None
        if self.use_features:
            grad_x_full, grad_y_full = compute_gradients(img_pad)

        nnf_y, nnf_x, nnf_cost, rot_idx, scale_idx = initialize_nnf(
            img_pad, mask_pad, h_c, w_c, self.patch_size, self.knn,
            self.rotations, self.scales, valid_sources)
        nnf.y, nnf.x, nnf.cost = nnf_y, nnf_x, nnf_cost
        nnf.rot_idx, nnf.scale_idx = rot_idx, scale_idx

        for level_idx in range(num_levels - 1, -1, -1):
            if self.verbose:
                print(f"   Level {level_idx} ({pyramid[level_idx].shape[:2]})")
            current_img = pyramid[level_idx]
            current_mask = mask_pyramid[level_idx]
            h_lvl, w_lvl = current_img.shape[:2]

            img_pad_lvl = np.pad(current_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
            mask_pad_lvl = np.pad(current_mask, ((pad, pad), (pad, pad)), mode='constant', constant_values=False)

            if self.use_features:
                grad_x_full, grad_y_full = compute_gradients(img_pad_lvl)
            else:
                grad_x_full, grad_y_full = None, None

            if level_idx < num_levels - 1:
                nnf = self._upsample_nnf(nnf, h_lvl, w_lvl)
                recompute_nnf_costs(img_pad_lvl, mask_pad_lvl, nnf.y, nnf.x, nnf.cost,
                                    nnf.rot_idx, nnf.scale_idx, h_lvl, w_lvl,
                                    self.patch_size, self.knn, self.rotations, self.scales,
                                    self.transform_maps, self.use_features, self.feature_weight,
                                    grad_x_full, grad_y_full)

            for it in range(self.iterations):
                if self.verbose and level_idx == 0:
                    print(f"      Iteration {it + 1}/{self.iterations}")
                propagate_forward(img_pad_lvl, mask_pad_lvl, nnf.y, nnf.x, nnf.cost,
                                  nnf.rot_idx, nnf.scale_idx, h_lvl, w_lvl, self.patch_size, self.knn,
                                  self.rotations, self.scales, self.transform_maps,
                                  self.use_features, self.feature_weight,
                                  grad_x_full, grad_y_full)
                propagate_backward(img_pad_lvl, mask_pad_lvl, nnf.y, nnf.x, nnf.cost,
                                   nnf.rot_idx, nnf.scale_idx, h_lvl, w_lvl, self.patch_size, self.knn,
                                   self.rotations, self.scales, self.transform_maps,
                                   self.use_features, self.feature_weight,
                                   grad_x_full, grad_y_full)
                random_search(img_pad_lvl, mask_pad_lvl, nnf.y, nnf.x, nnf.cost,
                              nnf.rot_idx, nnf.scale_idx, h_lvl, w_lvl, self.patch_size, self.knn,
                              self.rotations, self.scales, self.transform_maps,
                              max(h_lvl, w_lvl), self.random_decay,
                              self.use_features, self.feature_weight,
                              grad_x_full, grad_y_full)
                if self.use_coherence:
                    coherence_search(img_pad_lvl, mask_pad_lvl, nnf.y, nnf.x, nnf.cost,
                                     nnf.rot_idx, nnf.scale_idx, h_lvl, w_lvl, self.patch_size, self.knn,
                                     self.rotations, self.scales, self.transform_maps,
                                     self.use_features, self.feature_weight,
                                     grad_x_full, grad_y_full)
                if self.use_bidirectional and level_idx == 0:
                    bidirectional_heuristic(nnf.y, nnf.x, nnf.cost,
                                            nnf.rot_idx, nnf.scale_idx, h_lvl, w_lvl, self.knn)

        # Final reconstruction at full resolution
        final_img = image.astype(np.float32)
        final_mask = mask
        img_pad_final = np.pad(final_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        mask_pad_final = np.pad(final_mask, ((pad, pad), (pad, pad)), mode='constant', constant_values=False)

        if nnf.h != image.shape[0] or nnf.w != image.shape[1]:
            nnf = self._upsample_nnf(nnf, image.shape[0], image.shape[1])
            if self.use_features:
                grad_x_full, grad_y_full = compute_gradients(img_pad_final)
            else:
                grad_x_full, grad_y_full = None, None
            recompute_nnf_costs(img_pad_final, mask_pad_final, nnf.y, nnf.x, nnf.cost,
                                nnf.rot_idx, nnf.scale_idx, image.shape[0], image.shape[1],
                                self.patch_size, self.knn, self.rotations, self.scales,
                                self.transform_maps, self.use_features, self.feature_weight,
                                grad_x_full, grad_y_full)

        result = self._reconstruct_image_voting(img_pad_final, mask_pad_final,
                                                nnf.y, nnf.x, nnf.cost,
                                                nnf.rot_idx, nnf.scale_idx,
                                                image.shape[0], image.shape[1], self.knn,
                                                self.patch_size, self.transform_maps,
                                                self.rotations, self.scales,
                                                self.use_features, self.feature_weight,
                                                self.spatial_kernel)

        if self.verbose:
            print(f"   ✅ Done in {time.time() - start_time:.2f}s")
        return result

    # ============================================================================
    # RECONSTRUCTION (serial, stable weighting)
    # ============================================================================
    def _reconstruct_image_voting(self, img_pad, mask_pad, nnf_y, nnf_x, nnf_cost,
                                  rot_idx, scale_idx, h, w, k, patch_size,
                                  transform_maps, rotations, scales,
                                  use_features, feature_weight, spatial_kernel):
        pad = patch_size // 2
        c = img_pad.shape[2]
        result = np.zeros((h, w, c), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        maps_x, maps_y, w00, w01, w10, w11 = transform_maps
        n_rots, n_scales = len(rotations), len(scales)
        half = patch_size // 2

        from .core import extract_patch_bilinear, apply_cached_transform
        
        @njit(cache=True)
        def reconstruct(img_pad, mask_pad, nnf_y, nnf_x, nnf_cost,
                        rot_idx, scale_idx, h, w, k, patch_size,
                        maps_x, maps_y, w00, w01, w10, w11,
                        rotations, scales, n_rots, n_scales,
                        spatial_kernel):
            pad = patch_size // 2
            c = img_pad.shape[2]
            result = np.zeros((h, w, c), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)
            half = patch_size // 2

            for y in range(h):
                for x in range(w):
                    if not mask_pad[y + pad, x + pad]:
                        result[y, x] = img_pad[y + pad, x + pad]
                        weight_sum[y, x] = 1.0
                        continue

                    for ki in range(k):
                        sy = float(y) + nnf_y[y, x, ki]
                        sx = float(x) + nnf_x[y, x, ki]
                        sy_int, sx_int = int(round(sy)), int(round(sx))
                        if (0.0 <= sy <= float(h - 1) and 0.0 <= sx <= float(w - 1) and
                                0 <= sy_int < h and 0 <= sx_int < w and
                                not mask_pad[sy_int + pad, sx_int + pad]):
                            cost = nnf_cost[y, x, ki]
                            weight = 1.0 / (cost + 1e-4)
                            weight = min(max(weight, 1e-6), 1e6)
                            ri, si = rot_idx[y, x, ki], scale_idx[y, x, ki]
                            if ri < 0 or si < 0:
                                continue
                            tidx = ri * n_scales + si

                            src_patch = extract_patch_bilinear(img_pad, sy_int + pad, sx_int + pad, patch_size)
                            transformed = apply_cached_transform(src_patch,
                                                                 maps_x[tidx], maps_y[tidx],
                                                                 w00[tidx], w01[tidx], w10[tidx], w11[tidx])

                            for i in range(patch_size):
                                for j in range(patch_size):
                                    ty = y - half + i
                                    tx = x - half + j
                                    if 0 <= ty < h and 0 <= tx < w and mask_pad[ty + pad, tx + pad]:
                                        spatial_w = spatial_kernel[i, j]
                                        final_w = weight * spatial_w
                                        for ch in range(c):
                                            result[ty, tx, ch] += final_w * transformed[i, j, ch]
                                        weight_sum[ty, tx] += final_w

            for y in range(h):
                for x in range(w):
                    if weight_sum[y, x] > 0:
                        result[y, x] /= weight_sum[y, x]
                    else:
                        # fallback to neighbor average
                        sum_val = np.zeros(c, dtype=np.float32)
                        cnt = 0
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and weight_sum[ny, nx] > 0:
                                sum_val += result[ny, nx]
                                cnt += 1
                        result[y, x] = sum_val / cnt if cnt > 0 else 128.0

            return np.clip(result, 0, 255).astype(np.uint8)

        return reconstruct(img_pad, mask_pad, nnf_y, nnf_x, nnf_cost,
                           rot_idx, scale_idx, h, w, k, patch_size,
                           maps_x, maps_y, w00, w01, w10, w11,
                           rotations, scales, n_rots, n_scales,
                           spatial_kernel)