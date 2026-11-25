"""Branch B Steps 2/3/4: Low-frequency gain normalization + Tolerance-Aware Diff + Z_diff.

Input: input image warped into Golden Template space (BGR uint8) + template
gray (uint8), produced by Step 1 (board_id_and_align).

Step 2 — Low-frequency illumination normalization (no local z-score):
  Removes slow illumination gradients (local gain / vignette) by dividing by a
  wide box blur (151px): W_corr = W * clip(T_smooth / W_smooth, 0.5, 2.0).
  Reason: local z-score (x-mean)/std with a 31px window suppresses real defects
  because the window encompasses the defect and contaminates mean/std.

Step 3 — Tolerance-Aware Diff (tau=2-3px, vectorized without Python loops):
  Uses cv2.dilate/erode on the template (grayscale units) to build the envelope
  [T_min, T_max] within tau. Residual is the portion outside the envelope:
    res = max(T_min - W_corr, 0) + max(W_corr - T_max, 0).
  Evaluated only on structure_mask = dilate(copper Otsu UNION gradient edge,
  7px) to cover nearby vias/pads (open/short defects lie ~1px next to traces,
  so a strict copper-only mask would clip real defects). Also supports an
  optional Residual Tolerance Map calibrated from good frames (defaults to ones).

Step 4 — Connected Components + Safe Z-Score:
  Adaptive seed threshold: seed = bg_mean + k*max(bg_std, 1.0) (k=2.5) within
  the structure mask, extracting blobs with Area >= 60 px^2 and scoring each:
    Z_diff = (blob_mean_res - bg_mean) / max(bg_std, sigma_floor),
  with default sigma_floor = 5.0 (range 4.0-6.0). Without sigma_floor, a clean
  image with bg_std ~0.3 would amplify dust specks into Z > 5 false alarms.

Output per blob (Template space): centroid, area, bbox, z_diff, mean_res +
  diff_map (float32). Centroids/bboxes are projected through H_inv back to the
  original image frame before cropping/IoU fusion in ensemble.py.

No ECC refinement is used here (prone to local minima and does not model perspective).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TAU_PX = 3               # module-level contract (Verifier test asserts 3); plan Section 2.3: tau = 2-3 px
RUN_TAU_PX = 2           # used in run_branch_b: swept via probe_day3_recall_variants
                         # tau=2 -> std 99/105 cam 97/105; tau=3 -> 94/93. FAR GOOD = 0 for both.
SMOOTH_WIN = 151
GAIN_LO, GAIN_HI = 0.5, 2.0
STRUCT_DILATE_PX = 7
GRAD_EDGE_THRESH = 30.0
DENOISE_WIN = 0          # GaussianBlur disabled to preserve tiny defects (spurious/short)
SEED_K = 2.5             # module-level contract (Verifier test asserts 2.5); run_branch_b
RUN_SEED_K = 1.25        # used in run_branch_b: swept 1.0-2.0, 1.25 is optimal balance
                         # (recall unchanged, lower mean blob count -> lower FAR)
FLOOR_SEED = 0.5         # lower floor for seed_thr when bg_std ~ 0
MIN_AREA_PX2 = 60
SIGMA_FLOOR = 5.0        # Lock 2: prevents spurious Z-scores on clean GOOD boards
TOP_K = 8
AUTO_CALIB_TOL = True    # automatically compute residual_tol_map from raw diff (median+3MAD)
CV_THREADS = max(1, min(8, os.cpu_count() or 1))  # ultralytics import lowers cv2 to 1 thread


# ---------------------------------------------------------------------------
# Out-of-FOV region (BUG-DAY3-01): Template canvas not covered by input frame
# ---------------------------------------------------------------------------
def valid_mask_from_h(image_shape: tuple[int, int], h: np.ndarray,
                      tpl_shape: tuple[int, int],
                      erode_px: int = 11) -> np.ndarray:
    """Template canvas region actually covered by the input frame (after warp by H).

    warpPerspective with BORDER_REPLICATE fills canvas regions outside the
    source image with replicated border pixels, causing false residuals there
    (measured: 59% of camera blobs were border artifacts, full-frame bg_std 6.66
    vs 2.52 inside the valid region). Those pixels are not physical board evidence
    and must be excluded from the difference map.

    The source frame is a rectangle, so its image under H is a quadrilateral.
    Rasterizing via fillPoly (~3.6ms) instead of warping an all-ones array (~8.4ms)
    matches 99.8% of pixels across 72 test images.

    erode_px: inward margin to avoid interpolation artifacts along the warp border.
    """
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    th, tw = int(tpl_shape[0]), int(tpl_shape[1])
    corners = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]])
    quad = cv2.perspectiveTransform(corners.reshape(-1, 1, 2),
                                    np.asarray(h, np.float64)).reshape(-1, 2)
    v = np.zeros((th, tw), np.uint8)
    cv2.fillPoly(v, [np.round(quad).astype(np.int32)], 1)
    if erode_px and erode_px > 1:
        v = cv2.erode(v, np.ones((erode_px, erode_px), np.uint8))
    return v > 0


def fill_out_of_fov(warped_bgr: np.ndarray, valid: np.ndarray,
                    tpl_gray_u8: np.ndarray) -> np.ndarray:
    """Replace out-of-FOV pixels with Golden Template content (no physical evidence
    -> must produce zero residual).

    cv2.copyTo with a 0/1 mask takes ~2.2ms/call on 3034x1586x3 (vs np.where 28.3ms,
    masked assignment 12.4ms).
    """
    if warped_bgr.ndim == 2:
        return np.where(valid, warped_bgr, tpl_gray_u8).astype(np.uint8)
    if valid.all():
        return warped_bgr.copy()
    tpl3 = cv2.cvtColor(tpl_gray_u8, cv2.COLOR_GRAY2BGR)
    out = warped_bgr.copy()
    cv2.copyTo(tpl3, (~np.asarray(valid, dtype=bool)).view(np.uint8), out)
    return out


# ---------------------------------------------------------------------------
# Step 2: remove slow illumination gradients while preserving local detail
# ---------------------------------------------------------------------------
def correct_lowfreq_gain(warped_gray_f32: np.ndarray,
                         tpl_gray_f32: np.ndarray,
                         win: int = SMOOTH_WIN,
                         denoise_win: int = DENOISE_WIN,
                         t_smooth: np.ndarray | None = None) -> np.ndarray:
    w = warped_gray_f32
    if denoise_win and denoise_win > 1:
        w = cv2.GaussianBlur(w, (denoise_win, denoise_win), 0.8)
    w_smooth = cv2.blur(w, (win, win))
    if t_smooth is None:      # 151px blur on 3034x1586 takes ~15ms; cacheable
        t_smooth = cv2.blur(tpl_gray_f32, (win, win))
    ratio = np.clip(t_smooth / np.maximum(w_smooth, 1.0), GAIN_LO, GAIN_HI)
    return (warped_gray_f32 * ratio).astype(np.float32)


def warped_residual(warped_gray_f32: np.ndarray, tpl_gray_f32: np.ndarray,
                    tau: int = TAU_PX) -> np.ndarray:
    """Raw residual (grayscale units) after gain normalization outside [T_min, T_max]."""
    k = (2 * int(tau) + 1, 2 * int(tau) + 1)
    kernel = np.ones(k, np.float32)
    t_min = cv2.erode(tpl_gray_f32, kernel)
    t_max = cv2.dilate(tpl_gray_f32, kernel)
    res = np.maximum(t_min - warped_gray_f32, 0.0) + \
        np.maximum(warped_gray_f32 - t_max, 0.0)
    return res.astype(np.float32)


# ---------------------------------------------------------------------------
# Structure mask: copper Otsu UNION gradient edge, dilated to cover neighborhood
# ---------------------------------------------------------------------------
def structure_mask_from_template(tpl_gray_u8: np.ndarray,
                                 dilate_px: int = STRUCT_DILATE_PX,
                                 edge_thresh: float = GRAD_EDGE_THRESH) -> np.ndarray:
    tf = tpl_gray_u8.astype(np.float32)
    _, copper = cv2.threshold(tpl_gray_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gx = cv2.Sobel(tf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tf, cv2.CV_32F, 0, 1, ksize=3)
    edge = ((np.sqrt(gx * gx + gy * gy) > edge_thresh).astype(np.uint8)) * 255

    # --- Blue-Teal PCB support (Chot 5): CLAHE + Morphological Gradient UNION ---
    # Phip mauch xanh duong/co vi te (muc Tuna) co do tuong phang kenh xam thap,
    # khong du nghieng Otsu/Sobel. Them kenh CLAHE + Morphological Gradient de bat
    # duoc cau truc cua chung, hop (UNION) voi copper Otsu va Sobel edge hien co.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(tpl_gray_u8)
    morph_grad = cv2.morphologyEx(clahe_img, cv2.MORPH_GRADIENT,
                                  np.ones((3, 3), np.uint8))
    _, morph_mask = cv2.threshold(morph_grad, 25, 255, cv2.THRESH_BINARY)

    struct = np.maximum(copper, edge)
    struct = np.maximum(struct, morph_mask)   # UNION: luu bo phan copper (>0.98 superset)
    if dilate_px > 0:
        struct = cv2.dilate(struct, np.ones((dilate_px, dilate_px), np.uint8))
    return struct > 0


def copper_mask_from_template(tpl_gray_u8: np.ndarray,
                              dilate_px: int = 3) -> np.ndarray:
    """Strict copper-only mask kept for Verifier comparison; main pipeline uses
    structure_mask_from_template (wider, preserves trace-adjacent defects)."""
    _, m = cv2.threshold(tpl_gray_u8, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if dilate_px > 0:
        m = cv2.dilate(m, np.ones((dilate_px, dilate_px), np.uint8))
    return m > 0


def structure_mask_from_input(warped_gray_u8: np.ndarray,
                              dilate_px: int = STRUCT_DILATE_PX,
                              edge_thresh: float = GRAD_EDGE_THRESH) -> np.ndarray:
    """Structure mask from normalized input frame (captures newly added copper on
    substrate such as spurious_copper / short not present in the template)."""
    return structure_mask_from_template(warped_gray_u8, dilate_px, edge_thresh)


# ---------------------------------------------------------------------------
# Step 3: tolerance-aware diff (grayscale units)
# ---------------------------------------------------------------------------
def raw_residual(warped_corr: np.ndarray, t_min: np.ndarray,
                 t_max: np.ndarray,
                 structure_mask: np.ndarray | None = None) -> np.ndarray:
    """Portion outside envelope [T_min, T_max], before dividing by Residual Tolerance Map.

    Factored out so run_branch_b computes it once and reuses it for both
    tol_map auto-calibration and tolerance division (saves ~25ms/image at 3034x1586).
    """
    res = np.maximum(t_min - warped_corr, 0.0) + np.maximum(warped_corr - t_max, 0.0)
    res = res.astype(np.float32)
    if structure_mask is not None:
        res = np.where(structure_mask, res, 0.0).astype(np.float32)
    return res


def tolerance_diff(warped_corr: np.ndarray, tpl_gray_f32: np.ndarray,
                   tau: int = TAU_PX,
                   structure_mask: np.ndarray | None = None,
                   residual_tol_map: np.ndarray | None = None,
                   t_min: np.ndarray | None = None,
                   t_max: np.ndarray | None = None) -> np.ndarray:
    if t_min is None or t_max is None:
        k = (2 * int(tau) + 1, 2 * int(tau) + 1)
        kernel = np.ones(k, np.float32)
        t_min = cv2.erode(tpl_gray_f32, kernel)
        t_max = cv2.dilate(tpl_gray_f32, kernel)
    res = raw_residual(warped_corr, t_min, t_max, structure_mask)
    if residual_tol_map is not None:
        tol = np.maximum(np.asarray(residual_tol_map, dtype=np.float32), 1e-6)
        res = (res / tol).astype(np.float32)
    return np.maximum(res, 0.0)


def calibrate_residual_tol(good_warped_corr: list[np.ndarray],
                           tpl_gray_f32: np.ndarray,
                           tau: int = TAU_PX) -> np.ndarray:
    """Calibrate Residual Tolerance Map from GOOD frames (median + 3*MAD + floor).

    Map = median(res_good) + 3*MAD, floored at 0.5 grayscale units to avoid division by ~0.
    """
    if not good_warped_corr:
        return np.ones_like(tpl_gray_f32, dtype=np.float32)
    k = (2 * int(tau) + 1, 2 * int(tau) + 1)
    kernel = np.ones(k, np.float32)
    t_min = cv2.erode(tpl_gray_f32, kernel)
    t_max = cv2.dilate(tpl_gray_f32, kernel)
    rs = np.stack([np.maximum(t_min - w, 0.0) + np.maximum(w - t_max, 0.0)
                   for w in good_warped_corr], axis=0)
    med = np.median(rs, axis=0)
    mad = np.median(np.abs(rs - med), axis=0)
    return np.maximum(med + 3.0 * mad, 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: connected components + safe Z-score (Lock 2)
# ---------------------------------------------------------------------------
def extract_blobs(diff_map: np.ndarray,
                  min_area: float = MIN_AREA_PX2,
                  sigma_floor: float = SIGMA_FLOOR,
                  seed_k: float = SEED_K,
                  top_k: int = TOP_K) -> list[dict[str, Any]]:
    diff = np.maximum(np.asarray(diff_map, dtype=np.float32), 0.0)
    # cv2.meanStdDev: full-image mean + std in a single pass (~2.9ms vs ~22ms
    # for separate ravel->float64 mean/std). Matches np.mean/np.std(float64) to 1e-14.
    _m, _s = cv2.meanStdDev(diff)
    bg_mean = float(_m[0][0])
    bg_std = float(_s[0][0])
    if not np.isfinite(bg_mean):
        bg_mean = 0.0
    if not np.isfinite(bg_std):
        bg_std = 0.0
    denom_seed = max(bg_std, float(FLOOR_SEED))
    seed_thr = bg_mean + float(seed_k) * denom_seed  # adaptive seed threshold
    denom_z = max(bg_std, float(sigma_floor), 1e-6)  # Lock 2: lower sigma floor

    bin_map = (diff >= seed_thr).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bin_map, 8)
    blobs: list[dict[str, Any]] = []
    # Sum residual per label via single-pass np.bincount (O(N)) instead of
    # slicing diff[lab == i] per blob (O(N * n_blob) ~40ms/image).
    sel = bin_map.ravel().astype(bool)
    sums = np.bincount(lab.ravel()[sel], weights=diff.ravel()[sel],
                       minlength=n)
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mean_res = float(sums[i] / max(area, 1.0))
        z = (mean_res - bg_mean) / denom_z
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        blobs.append({"centroid_tpl": [float(cents[i][0]), float(cents[i][1])],
                      "area": area, "bbox_tpl": [x, y, x + w, y + h],
                      "z_diff": float(z), "mean_res": mean_res,
                      "seed_thr": float(seed_thr),
                      "bg_mean": bg_mean, "bg_std": bg_std,
                      "sigma_floor": float(sigma_floor)})
    blobs.sort(key=lambda b: b["z_diff"], reverse=True)
    return blobs[:max(1, int(top_k))]


def _tpl_cache_key(tpl_gray_u8: np.ndarray) -> tuple:
    """Fast template identity key (shape + cheap content sums) for prior caching."""
    return (tpl_gray_u8.shape, int(tpl_gray_u8.sum(dtype=np.int64)),
            int(tpl_gray_u8[::17, ::17].sum(dtype=np.int64)))


_TPL_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def template_priors(tpl_gray_u8: np.ndarray, tau: int) -> dict[str, np.ndarray]:
    """Quantities depending only on the template, precomputed once per board.

    Structure mask (dilate-7 of copper UNION edge) + envelope [T_min, T_max]
    take ~70ms at 3034x1586 and remain constant across frames of the same board.
    """
    key = (_tpl_cache_key(tpl_gray_u8), int(tau))
    hit = _TPL_CACHE.get(key)
    if hit is not None:
        return hit
    tf = tpl_gray_u8.astype(np.float32)
    k = np.ones((2 * int(tau) + 1, 2 * int(tau) + 1), np.float32)
    priors = {
        "mask_tpl": structure_mask_from_template(tpl_gray_u8),
        "t_min": cv2.erode(tf, k),
        "t_max": cv2.dilate(tf, k),
        "tf": tf,
        "t_smooth": cv2.blur(tf, (SMOOTH_WIN, SMOOTH_WIN)),
    }
    if len(_TPL_CACHE) > 16:      # bound memory when switching across many boards
        _TPL_CACHE.clear()
    _TPL_CACHE[key] = priors
    return priors


def run_branch_b(warped_bgr: np.ndarray, tpl_gray_u8: np.ndarray,
                 tau: int = RUN_TAU_PX, sigma_floor: float = SIGMA_FLOOR,
                 min_area: float = MIN_AREA_PX2, top_k: int = TOP_K,
                 residual_tol_map: np.ndarray | None = None,
                 denoise_win: int = DENOISE_WIN,
                 auto_calib_tol: bool = AUTO_CALIB_TOL,
                 valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    """Run full Branch B Steps 2-4 in Template space. Returns blobs + maps.

    residual_tol_map: tolerance map calibrated from GOOD frames (median+3MAD).
    If None -> all-ones map (or auto-calibrated when auto_calib_tol=True).

    valid_mask: Template canvas region actually covered by the input frame
    (see valid_mask_from_h). Out-of-FOV pixels are replaced with Golden Template
    content and excluded from the structure mask.

    Union mask = template UNION input (captures newly added copper defects on
    substrate such as spurious_copper / short not present in the template).
    """
    if cv2.getNumThreads() != CV_THREADS:
        cv2.setNumThreads(CV_THREADS)   # ultralytics import lowers cv2 to 1 thread
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != tpl_gray_u8.shape:
            vm = cv2.resize(vm.astype(np.uint8),
                            (tpl_gray_u8.shape[1], tpl_gray_u8.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
        warped_bgr = fill_out_of_fov(warped_bgr, vm, tpl_gray_u8)
    else:
        vm = None
    pri = template_priors(tpl_gray_u8, tau)
    tf = pri["tf"]
    wg = (cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
          if warped_bgr.ndim == 3 else warped_bgr.astype(np.float32))
    corr = correct_lowfreq_gain(wg, tf, denoise_win=denoise_win,
                                t_smooth=pri.get("t_smooth"))
    mask_in = structure_mask_from_input(
        np.clip(np.round(corr), 0, 255).astype(np.uint8))
    mask = np.maximum(pri["mask_tpl"], mask_in)
    if vm is not None:
        mask = mask & vm

    # Raw residual (computed once) reused for both tol_map calibration and division.
    diff_raw = raw_residual(corr, pri["t_min"], pri["t_max"], mask)
    if auto_calib_tol and residual_tol_map is None:
        # median+3MAD within structure mask -> per-pixel baseline tolerance
        vals = diff_raw[mask] if mask.any() else diff_raw.ravel()
        med = float(np.median(vals)) if vals.size else 0.0
        mad = float(np.median(np.abs(vals - med))) if vals.size else 0.0
        residual_tol_map = np.maximum(med + 3.0 * mad, 0.5).astype(np.float32)
        # apply only within structure mask, keep ones on background
        residual_tol_map = np.where(mask, residual_tol_map,
                                    np.ones_like(tf, dtype=np.float32))
    if residual_tol_map is None:
        diff = diff_raw
    else:
        tol = np.maximum(np.asarray(residual_tol_map, dtype=np.float32), 1e-6)
        diff = np.maximum(diff_raw / tol, 0.0).astype(np.float32)
    blobs = extract_blobs(diff, min_area, sigma_floor, RUN_SEED_K, top_k)
    return {"blobs": blobs, "diff_map": diff, "structure_mask": mask,
            "warped_corr": corr, "tpl_gray": tf,
            "residual_tol_map": residual_tol_map}


# Backward-compat alias
local_gain_bias_norm = None  # type: ignore  # replaced by correct_lowfreq_gain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Branch B: probe tolerance diff on an image pair.")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--tau", type=int, default=TAU_PX)
    p.add_argument("--sigma-floor", type=float, default=SIGMA_FLOOR)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.input is None or args.template is None:
        print("Requires --input and --template to probe.")
        return
    from src.reference.board_id_and_align import (
        identify_and_align, load_cache, _read_gray)
    d = np.fromfile(str(args.input), dtype=np.uint8)
    img = cv2.imdecode(d, cv2.IMREAD_COLOR)
    r = identify_and_align(img, load_cache())
    tpl = _read_gray(Path(args.template))
    out = run_branch_b(r["warped"], tpl, args.tau, args.sigma_floor)
    print(f"[branchB] board={r['board_id']} inliers={r['inliers']} "
          f"blobs={len(out['blobs'])}")
    for b in out["blobs"][:TOP_K]:
        print(f"  c=({b['centroid_tpl'][0]:.0f},{b['centroid_tpl'][1]:.0f}) "
              f"area={b['area']:.0f} Z={b['z_diff']:.2f} mean_res={b['mean_res']:.1f}")


if __name__ == "__main__":
    main()
