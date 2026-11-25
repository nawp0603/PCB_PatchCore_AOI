"""Day-3 — Branch B Step 1: Board ID & Coarse Alignment (ORB + RANSAC Homography).

Functionality:
  * Pre-extract ORB features of the 10 Golden Templates (assets/templates/,
    01/04/05-12.JPG) -> cache `.npz` in the new project (keypoints, descriptors,
    image sizes).
  * For an input image: match via BFMatcher + Lowe ratio test against each
    template, pick the board ID with the most inliers, estimate Homography H
    (Input -> Template) with cv2.findHomography(RANSAC) and H_inv = H^-1
    (Template -> Input).

Three anchors applied here:
  * Anchor 1 (Coordinate Sync): every blob found in Template space MUST be
    back-projected through H_inv (cv2.perspectiveTransform) into the original
    input image space before crop-640 and IoU. warp_to_template() +
    project_points() implement exactly these two directions.
  * No cv2.findTransformECC as the main path (it gets stuck in local minima
    on pad grid + cannot handle 8-DoF perspective). If ECC refinement is added
    later: inputMask=None or CV_8UC1 (avoids OpenCV 4.11 _mask.type() crash).
  * Zero-touch: templates read from assets/templates (inside project); .npz
    cache written inside the new project (via assert_writable_ok).

Terminology: 01/04 are "Zero-defect-training Boards" (they have a Golden
Sample but no defect data yet) — never called "unseen".
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.resource_paths import (  # noqa: E402
    LOCAL_TEMPLATES,
    NEW_PROJECT_DIR,
    TEMPLATE_FILES,
    assert_writable_ok,
    discover_templates,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.reference.tolerance_diff import (  # noqa: E402
    fill_out_of_fov,
    valid_mask_from_h,
)

DEFAULT_CACHE = NEW_PROJECT_DIR / "assets" / "templates" / "orb_cache.npz"
N_FEATURES = 8000
RATIO_TEST = 0.75
RANSAC_THRESH = 5.0
MIN_INLIERS_BOARD_ID = 30
TOP_PER_TEMPLATE_MATCHES = 4000
EARLY_EXIT_INLIERS = 60        # inliers >= threshold -> confident board match, early exit
                               # (measured across 80 stress images — shift 25px,
                               #  rot 2deg, gain 0.85-1.15, bias 12, vignette, noise:
                               #  correct board MIN 107 inliers, runner-up MAX 13)
FAST_TOP_MATCHES = 400         # cap matches fed into RANSAC in fast mode

# Shortlist via gradient thumbnail fingerprint (BUG-DAY3-02): knnMatch x10
# templates is the primary bottleneck (~120ms). Gradient fingerprint @48x96
# takes ~10ms and achieves top-1 = 100% across 92 bench images -> only 1-2
# templates need ORB matching.
SHORTLIST_TOP_K = 2            # number of candidate templates sent to ORB + RANSAC
SHORTLIST_H, SHORTLIST_W = 48, 96

# BUG-DAY3-02: `import ultralytics` calls cv2.setNumThreads(1) and sets
# OMP_NUM_THREADS=1 globally. Running alignment on 1 thread is ~4x slower
# (55.9ms on 16 threads vs 215ms on 1 thread). Restore thread count before
# each alignment call.
CV_THREADS = max(1, min(8, os.cpu_count() or 1))

# 8000 template keypoints take ~18ms/template in knnMatch. Subsampling 3000
# evenly spaced descriptors (linspace) gives optimal Branch B recall across
# 72 std+cam images and reduces alignment latency from 54.4ms to 41.4ms.
FAST_TPL_MAX_DESC = 3000

# FLANN LSH parameters for ORB descriptors (CV_8U required; BFMatcher is primary).
FLANN_INDEX_LSH = 6
FLANN_PARAMS = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6, key_size=12, multi_probe_level=1)
FLANN_SEARCH_PARAMS = dict(checks=32)


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------
def _orb() -> cv2.ORB:
    return cv2.ORB_create(nfeatures=N_FEATURES)


def _read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def build_cache(cache_path: Path = DEFAULT_CACHE,
                n_features: int = N_FEATURES,
                template_files: list[str] | None = None) -> dict[str, Any]:
    """Extract ORB features for each template and save compressed .npz cache.

    template_files: list of template filenames to index. If None -> uses
    discover_templates() (includes the 10 standard boards 01/04/05-12 plus
    any newly enrolled templates).
    """
    cache_path = assert_writable_ok(Path(cache_path))
    if template_files is None:
        template_files = discover_templates()
    orb = cv2.ORB_create(nfeatures=n_features)
    out: dict[str, Any] = {"board_ids": [], "shapes": []}
    descs: list[np.ndarray] = []
    kps_xy: list[np.ndarray] = []
    for fname in template_files:
        bid = Path(fname).stem
        img = _read_gray(LOCAL_TEMPLATES / fname)
        kp, des = orb.detectAndCompute(img, None)
        if des is None or len(kp) == 0:
            raise RuntimeError(f"ORB found no keypoints: {fname}")
        out["board_ids"].append(bid)
        out["shapes"].append(list(img.shape))
        kps_xy.append(np.float32([p.pt for p in kp]))
        descs.append(des.astype(np.uint8))
    n = [len(k) for k in kps_xy]
    max_kp = max(n)
    kp_pad = np.zeros((len(kps_xy), max_kp, 2), np.float32)
    des_pad = np.zeros((len(descs), max_kp, 32), np.uint8)
    for i, (k, d) in enumerate(zip(kps_xy, descs)):
        kp_pad[i, :len(k)] = k
        des_pad[i, :len(d)] = d
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, board_ids=np.array(out["board_ids"]),
        shapes=np.array(out["shapes"], np.int32), n_kp=np.array(n, np.int32),
        keypoints=kp_pad, descriptors=des_pad)
    out.update({"cache": str(cache_path), "n_kp": n})
    for b, c in zip(out["board_ids"], n):
        print(f"  [cache] board {b}: {c} keypoints")
    print(f"[cache] wrote {cache_path}")
    return out


def _tpl_image_path(board_id: str) -> Path | None:
    """Resolve template image path for board_id (.JPG/.jpg/.jpeg/.png).

    Returns None if not found in LOCAL_TEMPLATES.
    """
    for ext in (".JPG", ".jpg", ".jpeg", ".png"):
        p = LOCAL_TEMPLATES / f"{board_id}{ext}"
        if p.exists():
            return p
    return None


def load_cache(cache_path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    z = np.load(str(cache_path), allow_pickle=False)
    board_ids = [str(b) for b in z["board_ids"]]
    n_kp = [int(v) for v in z["n_kp"]]
    kps, des = z["keypoints"], z["descriptors"]
    kps_list = [np.ascontiguousarray(kps[i, :n_kp[i]]) for i in range(len(board_ids))]
    des_list = [np.ascontiguousarray(des[i, :n_kp[i]]) for i in range(len(board_ids))]
    # FLANN LSH indexer per template descriptor (fallback if needed).
    # Descriptors must remain CV_8U as LSH does not accept float32.
    flann_indexers: list[cv2.FlannBasedMatcher | None] = []
    for d in des_list:
        try:
            fm = cv2.FlannBasedMatcher(FLANN_PARAMS, FLANN_SEARCH_PARAMS)
            fm.add([d.astype(np.uint8)])
            fm.train()
            flann_indexers.append(fm)
        except cv2.error:
            flann_indexers.append(None)
    # Subsampled descriptors for fast matching (see FAST_TPL_MAX_DESC).
    kps_fast, des_fast = [], []
    for k, d in zip(kps_list, des_list):
        if len(d) > FAST_TPL_MAX_DESC:
            sel = np.linspace(0, len(d) - 1, FAST_TPL_MAX_DESC).astype(np.int32)
            kps_fast.append(np.ascontiguousarray(k[sel]))
            des_fast.append(np.ascontiguousarray(d[sel]))
        else:
            kps_fast.append(k)
            des_fast.append(d)
    # Gradient thumbnail fingerprints + grayscale templates decoded once here.
    tpl_grays = []
    for b in board_ids:
        p = _tpl_image_path(b)
        if p is None:
            raise FileNotFoundError(f"Template image not found for board {b} "
                                    f"in {LOCAL_TEMPLATES}")
        tpl_grays.append(_read_gray(p))
    fps = np.stack([fingerprint(g) for g in tpl_grays], axis=0)
    return {"board_ids": board_ids, "shapes": [tuple(s) for s in z["shapes"]],
            "n_kp": n_kp, "keypoints": kps_list, "descriptors": des_list,
            "keypoints_fast": kps_fast, "descriptors_fast": des_fast,
            "flann_indexers": flann_indexers, "fingerprints": fps,
            "templates_gray": tpl_grays}


# ---------------------------------------------------------------------------
# Shortlist: gradient thumbnail fingerprint (invariant to gain/bias/vignette)
# ---------------------------------------------------------------------------
def fingerprint(gray: np.ndarray,
                th: tuple[int, int] = (SHORTLIST_H, SHORTLIST_W)) -> np.ndarray:
    """Normalized 48x96 Sobel gradient feature vector (cosine similarity ~ NCC).

    Uses gradient magnitude rather than raw intensity to suppress camera
    gain/bias/vignette variations. Achieves 100% top-1 accuracy on 92 bench images.
    """
    t = cv2.resize(gray, (int(th[1]), int(th[0])),
                   interpolation=cv2.INTER_LINEAR)
    f = t.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    f = np.sqrt(gx * gx + gy * gy)
    f = (f - float(f.mean())) / max(float(f.std()), 1e-6)
    return f.ravel().astype(np.float32)


def shortlist_boards(gray: np.ndarray, fingerprints: np.ndarray,
                     board_ids: list[str],
                     top_k: int = SHORTLIST_TOP_K) -> list[tuple[str, float]]:
    """Rank templates by cosine similarity against the input gradient fingerprint."""
    fp = fingerprint(gray)
    n = np.linalg.norm(fp)
    t = np.linalg.norm(fingerprints, axis=1)
    cos = (fingerprints @ fp) / np.maximum(t * max(float(n), 1e-9), 1e-9)
    order = np.argsort(-cos)[:max(1, int(top_k))]
    return [(board_ids[i], float(cos[i])) for i in order]


# ---------------------------------------------------------------------------
# Board ID + H / H_inv
# ---------------------------------------------------------------------------
def match_template(input_des: np.ndarray, tpl_des: np.ndarray,
                   ratio: float = RATIO_TEST,
                   flann_indexer: cv2.FlannBasedMatcher | None = None,
                   flann_k: int = 2) -> list[cv2.DMatch]:
    if input_des is None or tpl_des is None or len(input_des) < 2 or len(tpl_des) < 2:
        return []
    if flann_indexer is not None:
        try:
            # kNN via FLANN LSH; LSH requires CV_8U descriptors
            matches = flann_indexer.knnMatch(np.ascontiguousarray(input_des, np.uint8),
                                             k=flann_k)
            good: list[cv2.DMatch] = []
            for pair in matches:
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                    good.append(pair[0])
            good.sort(key=lambda m: m.distance)
            return good[:TOP_PER_TEMPLATE_MATCHES]
        except cv2.error:
            pass  # fallback to BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(input_des, tpl_des, k=2)
    good: list[cv2.DMatch] = []
    for p in pairs:
        if len(p) == 2 and p[0].distance < ratio * p[1].distance:
            good.append(p[0])
    good.sort(key=lambda m: m.distance)
    return good[:TOP_PER_TEMPLATE_MATCHES]


def estimate_h(kp_in: np.ndarray, kp_tpl: np.ndarray,
               matches: list[cv2.DMatch],
               ransac_thresh: float = RANSAC_THRESH) -> tuple[np.ndarray | None, int]:
    """Estimate Homography H: Input -> Template (RANSAC). Returns (H | None, n_inliers)."""
    if len(matches) < 4:
        return None, 0
    src = np.float32([kp_in[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_tpl[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if h is None or mask is None:
        return None, 0
    return h.astype(np.float64), int(mask.sum())


def _match_one_board(bid: str, des_in: np.ndarray, kp_in: np.ndarray,
                     cache: dict[str, Any], ransac_thresh: float,
                     best: dict[str, Any]) -> int:
    """ORB-match a single template in fast mode and update `best` in place.

    Shared by both initial shortlist matching and fallback escalation (BUG-DAY3-04).
    Returns the inlier count for this template.
    """
    idx = cache["board_ids"].index(bid)
    kt = np.ascontiguousarray(cache["keypoints_fast"][idx])
    dt = np.ascontiguousarray(cache["descriptors_fast"][idx])
    shape = cache["shapes"][idx]
    m = match_template(des_in, dt)
    h, inl = estimate_h(kp_in, kt, m[:FAST_TOP_MATCHES], ransac_thresh)
    best["scores"][bid] = inl
    if inl > best["inliers"]:
        best.update(board_id=bid, H=h, inliers=inl, tpl_shape=tuple(shape))
    return inl


def identify_and_align(image_bgr: np.ndarray, cache: dict[str, Any] | None = None,
                       cache_path: Path = DEFAULT_CACHE,
                       ransac_thresh: float = RANSAC_THRESH,
                       fast: bool = False,
                       fast_nfeatures: int = 1200,
                       fast_scale: float = 0.5,
                       fast_topk_h: int = SHORTLIST_TOP_K,
                       pinned_board_id: str | None = None) -> dict[str, Any]:
    """Identify board ID, estimate H / H_inv, and warp input into Template space.

    Returns dict: board_id, H (Input->Template), H_inv (Template->Input),
    inliers, warped (input warped to Template space with out-of-FOV filled, BGR uint8),
    tpl_shape, scores {bid: inliers}, mode ("full"/"fast").

    fast=True: detect ORB on half-resolution image + shortlist via gradient
    fingerprint, running RANSAC on top candidates (~45ms/image).

    pinned_board_id: if provided and present in cache["board_ids"], skips
    shortlist and directly matches the pinned board (~15-25ms).
    """
    if cache is None:
        cache = load_cache(cache_path)
    if cv2.getNumThreads() != CV_THREADS:
        # Restore OpenCV thread count after ultralytics import side-effect
        cv2.setNumThreads(CV_THREADS)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    if fast:
        small = cv2.resize(gray, None, fx=fast_scale, fy=fast_scale,
                           interpolation=cv2.INTER_AREA)
        orb = cv2.ORB_create(nfeatures=fast_nfeatures)
        kp_in_list, des_in = orb.detectAndCompute(small, None)
        scale_back = 1.0 / fast_scale
    else:
        orb = _orb()
        kp_in_list, des_in = orb.detectAndCompute(gray, None)
        scale_back = 1.0
    if des_in is None or not kp_in_list:
        raise RuntimeError("ORB found no keypoints on the input image")
    kp_in = np.float32([p.pt for p in kp_in_list]) * scale_back

    best: dict[str, Any] = {"board_id": None, "H": None, "inliers": -1,
                            "scores": {}, "shortlist_escalated": False}
    if pinned_board_id is not None and pinned_board_id in cache["board_ids"]:
        # Pinned board (Fixed Jig): match single template directly without shortlist.
        best["scores"][pinned_board_id] = 0
        if fast:
            inl = _match_one_board(pinned_board_id, des_in, kp_in, cache,
                                   ransac_thresh, best)
        else:
            # Full descriptors for maximum accuracy when fast=False.
            idx = cache["board_ids"].index(pinned_board_id)
            kt = np.ascontiguousarray(cache["keypoints"][idx])
            dt = np.ascontiguousarray(cache["descriptors"][idx])
            m = match_template(des_in, dt,
                               flann_indexer=cache["flann_indexers"][idx])
            h, inl = estimate_h(kp_in, kt, m, ransac_thresh)
            shape = cache["shapes"][idx]
            best["scores"][pinned_board_id] = inl
            if inl > best["inliers"]:
                best.update(board_id=pinned_board_id, H=h, inliers=inl,
                            tpl_shape=tuple(shape))
        best["scores"][pinned_board_id] = inl
        best["mode"] = "fast" if fast else "full"
        best["pinned"] = True
    elif fast:
        # Shortlist via gradient fingerprint -> ORB-match top-K templates first.
        # Escalates to remaining templates in cosine order if inliers < EARLY_EXIT_INLIERS.
        cand = shortlist_boards(gray, cache["fingerprints"], cache["board_ids"])
        for bid, cos in cand:
            best["scores"][bid] = round(cos, 4)
        order = [b for b, _c in cand]
        for bid in order:
            _match_one_board(bid, des_in, kp_in, cache, ransac_thresh, best)
            if best["inliers"] >= EARLY_EXIT_INLIERS:
                break
        if best["inliers"] < EARLY_EXIT_INLIERS:
            for bid in [b for b in cache["board_ids"] if b not in best["scores"]]:
                _match_one_board(bid, des_in, kp_in, cache, ransac_thresh, best)
                if best["inliers"] >= EARLY_EXIT_INLIERS:
                    break
            best["shortlist_escalated"] = True
    else:
        for i, (bid, kt, dt, shape) in enumerate(zip(cache["board_ids"], cache["keypoints"],
                                      cache["descriptors"], cache["shapes"])):
            fi = cache["flann_indexers"][i]
            m = match_template(des_in, np.ascontiguousarray(dt), flann_indexer=fi)
            h, inl = estimate_h(kp_in, np.ascontiguousarray(kt), m, ransac_thresh)
            best["scores"][bid] = inl
            if inl > best["inliers"]:
                best.update(board_id=bid, H=h, inliers=inl, tpl_shape=tuple(shape))
    best["mode"] = "fast" if fast else "full"
    if best["H"] is None or best["inliers"] < MIN_INLIERS_BOARD_ID:
        raise RuntimeError(f"Board identification failed (best={best['board_id']}, "
                           f"inliers={best['inliers']}). scores={best['scores']}")
    h = best["H"]
    try:
        h_inv = np.linalg.inv(h)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Degenerate homography H, cannot invert: {e}")
    th, tw = best["tpl_shape"]
    warped = cv2.warpPerspective(image_bgr, h, (tw, th),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    # BUG-DAY3-01: Replace out-of-FOV replicated border pixels with Golden Template
    # content so regions outside the camera FOV produce zero residual.
    valid = valid_mask_from_h(image_bgr.shape[:2], h, (th, tw))
    tpl_gray = cache["templates_gray"][cache["board_ids"].index(best["board_id"])]
    if tpl_gray.shape != (th, tw):
        tpl_gray = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
    warped_filled = fill_out_of_fov(warped, valid, tpl_gray)
    best["H_inv"] = h_inv
    best["warped"] = warped_filled       # out-of-FOV filled (used for diff)
    best["warped_raw"] = warped          # raw warp (available for UI display)
    best["valid_mask"] = valid
    return best


def _valid_board_id(board_id: str) -> str:
    """Validate that board_id contains only letters, digits, '_' and '-'."""
    bid = str(board_id).strip()
    if not bid or not re.fullmatch(r"[A-Za-z0-9_-]+", bid):
        raise ValueError(f"Invalid board_id: '{board_id}'. "
                         "Only letters, digits, underscore (_) and hyphen (-) allowed.")
    return bid


def enroll_template(image_bgr: np.ndarray, board_id: str,
                    cache_path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    """Enroll a Live Golden Template and hot-reload the ORB cache.

    1. Validate board_id (alphanumeric, underscore, hyphen).
    2. Save image_bgr to LOCAL_TEMPLATES / f"{board_id}.JPG" (via assert_writable_ok).
    3. Rebuild ORB cache on disk and return reloaded cache dict.
    """
    bid = _valid_board_id(board_id)
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image_bgr must be an ndarray (H, W, 3), BGR uint8")
    dst = assert_writable_ok(LOCAL_TEMPLATES / f"{bid}.JPG")
    ok = cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed to write template image: {dst}")
    # Hot-reload: rebuild cache including newly enrolled template and reload into RAM.
    build_cache(cache_path=cache_path)
    return load_cache(cache_path)


def project_points(pts_xy: np.ndarray, h_inv: np.ndarray) -> np.ndarray:
    """Lock 1: Project Template (x, y) -> original Input space via H_inv."""
    arr = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, h_inv.astype(np.float64)).reshape(-1, 2)


def warp_to_template(image_bgr: np.ndarray, h: np.ndarray,
                     tpl_shape: tuple[int, int]) -> np.ndarray:
    th, tw = int(tpl_shape[0]), int(tpl_shape[1])
    return cv2.warpPerspective(image_bgr, h, (tw, th),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Branch B Step 1: build ORB cache and test alignment.")
    p.add_argument("--build-cache", action="store_true")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--image", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.build_cache or args.image is None:
        build_cache(args.cache)
    if args.image is not None:
        cache = load_cache(args.cache)
        data = np.fromfile(str(args.image), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot decode image: {args.image}")
        r = identify_and_align(img, cache)
        print(f"[align] board={r['board_id']} inliers={r['inliers']} scores={r['scores']}")
        print(f"[align] H|H_inv ok, warped={r['warped'].shape} tpl={r['tpl_shape']}")


if __name__ == "__main__":
    main()
