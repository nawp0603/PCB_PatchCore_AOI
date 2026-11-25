"""Step 1 — Dual-Path Industrial AOI Engine (decoupled from any UI).

Wraps the full Branch A (YOLO Full-Res) + Branch B (ORB Align +
Tolerance Diff + Box Fusion) pipeline in an independent
`DualPathAOIEngine` class supporting:
  * Fixed Jig ROI Crop: cuts the fixture frame before processing, then
    shifts coordinates back to the original camera image system.
  * Pinned board ID (`pinned_board_id`): skips shortlist, direct ORB + RANSAC.
  * Live Golden Template Enrollment + Hot-Reload.
  * Safe fallback to full-board YOLO when Branch B cannot align
    (never crashes during live demos).
  * GPU warm-up at startup to remove first-run latency.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from configs.resource_paths import (  # noqa: E402
    LOCAL_TEMPLATES, default_weights,
)
import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402
from src.decision.logic import decide  # noqa: E402
from src.models.ensemble import fuse_and_decide  # noqa: E402
from src.models.yolo_detector import (  # noqa: E402
    CENTER_GATE_RADIUS, DEFAULT_CONF, DEFAULT_IMGSZ, YOLODetector,
)

# Bounding-box colors per class (BGR, same as src/ui/app.py).
# Mapped to PKU class IDs: 0=missing_hole, 1=mouse_bite, 2=open_circuit,
# 3=short, 4=spur, 5=spurious_copper — each semantic class keeps its
# traditional color regardless of ID position.
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),        # missing_hole — green
    1: (0, 255, 255),      # mouse_bite — yellow
    2: (0, 0, 255),        # open_circuit — red
    3: (0, 165, 255),      # short — orange
    4: (255, 0, 255),      # spur — purple
    5: (255, 100, 0),      # spurious_copper — blue
}

# ---------------------------------------------------------------------------
# Robot action mapping (4 levels -> sort bins; kept for API/test compatibility)
# ---------------------------------------------------------------------------
LEVEL_TO_ROBOT: dict[int, dict[str, Any]] = {
    1: {"should_pick": True, "bin_id": 1, "target_bin": "BIN_FAIL",
        "label": "Bin 1 - Critical Scrap (FAIL)"},
    2: {"should_pick": True, "bin_id": 2, "target_bin": "BIN_REWORK",
        "label": "Bin 2 - Repairable (REWORK)"},
    3: {"should_pick": True, "bin_id": 3, "target_bin": "BIN_REVIEW",
        "label": "Bin 3 - Operator Check (REVIEW)"},
    4: {"should_pick": True, "bin_id": 4, "target_bin": "BIN_PASS",
        "label": "Bin 4 - Good Board (PASS)"},
}

INPUT_MIN_SIDE = 640   # minimum short side for stable ORB + RANSAC (same as UI)


class DualPathAOIEngine:
    """Core inference engine for the Dual-Path AOI system.

    Portable across machines: YOLO device/imgsz auto-adapts to the local
    GPU (discrete NVIDIA -> CUDA FP16 full-res; low-VRAM -> capped 2560/1920;
    integrated graphics / no CUDA -> CPU FP32 1920). If CUDA init fails
    (driver mismatch after copying .venv), the engine retries on CPU
    instead of staying OFFLINE.
    """

    def __init__(self, weights_path: Path | None = None,
                 device: str | None = None,
                 imgsz: int | None = None) -> None:
        if weights_path is not None:
            self.weights_path = Path(weights_path)
        else:
            # Portable default: fine-tuned day2 if bundled, else base.
            # (Do NOT use the import-time DEFAULT_WEIGHTS constant directly:
            # on a fresh machine only one of the two files may exist.)
            self.weights_path = Path(default_weights())
        if not self.weights_path.exists():
            # Last-chance fallback to the other bundled weights file.
            alt = Path(default_weights())
            if alt.exists():
                self.weights_path = alt
            else:
                raise FileNotFoundError(
                    f"YOLO weights not found: {self.weights_path} "
                    f"(also tried {alt}; re-extract NET_DualPATH.zip)")
        self.device = device
        self.imgsz = imgsz if imgsz else DEFAULT_IMGSZ
        try:
            self.yolo = YOLODetector(self.weights_path, conf_threshold=DEFAULT_CONF,
                                     imgsz=self.imgsz, device=self.device)
        except Exception as first_err:
            # Cross-GPU safety net: a CUDA-built torch copied from another PC,
            # a broken NVIDIA driver, or a missing/foreign GPU (discrete /
            # integrated / none) can raise at init time. Retry once explicitly
            # on CPU so the station degrades instead of going OFFLINE.
            # (Retry on ANY first failure when device was auto/requested-cuda,
            # not only cuda-keyword messages, because torch/OOM/driver errors
            # vary across machines.)
            already_cpu = (device is not None
                           and str(device).lower() == "cpu")
            if not already_cpu:
                print(f"[DualPathAOIEngine] {first_err!r} -> retry on CPU "
                      f"(imgsz 1920, FP32).")
                try:
                    self.yolo = YOLODetector(
                        self.weights_path, conf_threshold=DEFAULT_CONF,
                        imgsz=min(int(self.imgsz), 1920), device="cpu")
                except Exception as cpu_err:
                    raise cpu_err from first_err
                try:
                    self.yolo.adapt_note = (
                        f"{getattr(self.yolo, 'adapt_note', '')} "
                        f"[recovered from init failure: {first_err}]").strip()
                except Exception:
                    pass
            else:
                raise
        # ORB cache (auto-built if missing).
        cache_path = ba.DEFAULT_CACHE
        if not Path(cache_path).exists():
            ba.build_cache(cache_path=cache_path)
        self.cache = ba.load_cache(cache_path)
        self._tpl_gray_cache: dict[tuple[str, tuple[int, int]], np.ndarray] = {}
        self._warmup_ms: float | None = None

    # ------------------------------------------------------------------ #
    # GPU warm-up (eliminates first-inference CUDA latency)
    # ------------------------------------------------------------------ #
    def warmup(self) -> float:
        """Run a dummy 640x640 black-image inference to initialize the CUDA context.

        Returns warm-up duration (ms) and stores it in self._warmup_ms.
        """
        black = np.zeros((640, 640, 3), np.uint8)
        if cv2.getNumThreads() != ba.CV_THREADS:
            cv2.setNumThreads(ba.CV_THREADS)
        t0 = time.perf_counter()
        try:
            self.yolo.predict_batch([black], conf=DEFAULT_CONF, imgsz=640)
        except Exception as warm_err:
            if getattr(self.yolo, "device", "cpu") == "cuda":
                print(f"[DualPathAOIEngine] CUDA warmup failed ({warm_err!r}) "
                      f"-> switching to CPU (FP32, imgsz 1920).")
                self.yolo.device = "cpu"
                self.yolo.half = False
                self.yolo.imgsz = min(int(getattr(self.yolo, "imgsz", 1920)), 1920)
                self.yolo.adapt_note = (
                    f"{getattr(self.yolo, 'adapt_note', '')} "
                    f"[CUDA warmup failed -> CPU FP32]").strip()
                try:
                    self.yolo.predict_batch([black], conf=DEFAULT_CONF, imgsz=640)
                except Exception:
                    pass
        try:
            # Pre-warm template priors for board "01" for fast first inspection
            if "01" in self.cache.get("board_ids", []):
                idx = list(self.cache["board_ids"]).index("01")
                sh = tuple(int(v) for v in self.cache["shapes"][idx])
                tpl01 = self._template_gray("01", sh)
                td.template_priors(tpl01, td.RUN_TAU_PX)
        except Exception:
            # Warm-up failure must not crash engine startup.
            pass
        self._warmup_ms = (time.perf_counter() - t0) * 1000.0
        return self._warmup_ms

    # ------------------------------------------------------------------ #
    # Template management
    # ------------------------------------------------------------------ #
    def reload_templates(self) -> list[str]:
        """Reload ORB cache from disk and clear tolerance_diff _TPL_CACHE.

        Returns the list of board_ids currently in cache.
        """
        self.cache = ba.load_cache(ba.DEFAULT_CACHE)
        self._tpl_gray_cache.clear()
        try:
            td._TPL_CACHE.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        return list(self.cache["board_ids"])

    def enroll_golden_template(self, image_bgr: np.ndarray, board_id: str,
                               roi_rect: list[int] | None = None) -> dict[str, Any]:
        """Enroll a Live Golden Template and hot-reload ORB cache.

        If a valid roi_rect is provided, crops the Jig region first, saves the
        template, and invokes enroll_template() (build_cache + hot-reload RAM).
        Returns enrolled metadata: board_id, n_keypoints, shape, saved path.
        """
        img = self._validate_bgr(image_bgr)
        if roi_rect is not None:
            img = self._crop_roi(img, roi_rect)[0]
        cache = ba.enroll_template(img, board_id, cache_path=ba.DEFAULT_CACHE)
        self.cache = cache
        # Extract n_keypoints + shape from updated cache.
        bid = board_id.strip()
        if bid in cache["board_ids"]:
            idx = cache["board_ids"].index(bid)
            n_keypoints = int(cache["n_kp"][idx])
            shape = tuple(int(v) for v in cache["shapes"][idx])
        else:
            n_keypoints, shape = 0, (0, 0)
        return {"board_id": bid, "n_keypoints": n_keypoints, "shape": shape,
                "saved": str(LOCAL_TEMPLATES / f"{bid}.JPG")}

    # ------------------------------------------------------------------ #
    # Core inspect (Dual-Path + Jig ROI + YOLO fallback + Robot mapping)
    # ------------------------------------------------------------------ #
    def inspect(self, image_bgr: np.ndarray, fast_align: bool = True,
                pinned_board_id: str | None = None,
                roi_rect: list[int] | None = None,
                allow_yolo_fallback: bool = True,
                include_visuals: bool = True) -> dict[str, Any]:
        """Execute the full Dual-Path inspection pipeline on a BGR uint8 image.

        Returns a unified result dict:
          status, board_id, inliers, alignment_mode, level, conclusion,
          verdict, reasons, robot_action, items, timings_ms,
          annotated_bgr, heatmap_bgr, raw_bgr.
        """
        img = self._validate_bgr(image_bgr)
        h_img, w_img = img.shape[:2]
        t_total = time.perf_counter()

        # Warn (not crash) when the pinned Board ID is unknown: Branch B
        # will fall through to full matching / YOLO fallback instead.
        pinned_unknown = (
            pinned_board_id is not None
            and str(pinned_board_id).strip() not in list(self.cache.get("board_ids", []))
        )

        # --- Step 1: Fixed Jig ROI crop --------------------------------
        offset = (0, 0)
        work_img = img
        if roi_rect is not None:
            work_img, offset = self._crop_roi(img, roi_rect)

        # --- Step 2: Branch B (align + tolerance diff) -----------------
        timings: dict[str, float] = {"align": 0.0, "diff": 0.0,
                                     "roi_yolo": 0.0, "total": 0.0}
        res: dict[str, Any] | None = None
        alignment_mode = "orb_align"
        board_id: str | None = None
        inliers: int | None = None
        blobs: list[dict[str, Any]] = []
        h_inv: np.ndarray | None = None
        diff_map: np.ndarray | None = None
        warped_shape: tuple[int, int] | None = None

        if cv2.getNumThreads() != ba.CV_THREADS:
            cv2.setNumThreads(ba.CV_THREADS)

        try:
            t0 = time.perf_counter()
            r = ba.identify_and_align(
                work_img, cache=self.cache, fast=fast_align,
                pinned_board_id=pinned_board_id)
            t_align = (time.perf_counter() - t0) * 1000.0
            timings["align"] = t_align
            board_id = r["board_id"]
            inliers = int(r["inliers"])
            alignment_mode = r.get("mode", "orb_align")
            h_inv = r["H_inv"]
            warped_shape = r["warped"].shape[:2]

            tpl = self._template_gray(board_id, warped_shape)
            t0 = time.perf_counter()
            out = td.run_branch_b(r["warped"], tpl,
                                  valid_mask=r.get("valid_mask"))
            t_diff = (time.perf_counter() - t0) * 1000.0
            timings["diff"] = t_diff
            blobs = out["blobs"]
            diff_map = out["diff_map"]

            # Lock 1: project blob centroids through H_inv (Template -> work_img).
            if blobs:
                cents = ba.project_points(
                    np.float32([b["centroid_tpl"] for b in blobs]), h_inv)
            else:
                cents = np.zeros((0, 2), np.float32)

            # --- Step 3: Branch A (batch 640x640 crops + predict_batch) -
            crops, origins = [], []
            w_work, h_work = work_img.shape[1], work_img.shape[0]
            for (px, py) in cents:
                x0 = int(np.clip(float(px) - 320, 0, max(w_work - 640, 0)))
                y0 = int(np.clip(float(py) - 320, 0, max(h_work - 640, 0)))
                x1, y1 = min(x0 + 640, w_work), min(y0 + 640, h_work)
                crop = work_img[y0:y1, x0:x1]
                if crop.shape[0] != 640 or crop.shape[1] != 640:
                    pad = np.zeros((640, 640, 3), np.uint8)
                    pad[:crop.shape[0], :crop.shape[1]] = crop
                    crop = pad
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                origins.append((x0, y0))

            t0 = time.perf_counter()
            batch = self.yolo.predict_batch(crops) if crops else []
            t_roi = (time.perf_counter() - t0) * 1000.0
            timings["roi_yolo"] = t_roi

            yolo_dets_all: list[dict[str, Any]] = []
            for dets_one, (px, py), (x0, y0) in zip(batch, cents, origins):
                kept = YOLODetector.spatial_center_gating(
                    dets_one, center_xy=(float(px), float(py)),
                    radius=float(CENTER_GATE_RADIUS),
                    crop_origin=(float(x0), float(y0)))
                for d in kept:
                    dd = dict(d)
                    dd["bbox"] = [d["bbox"][0] + x0, d["bbox"][1] + y0,
                                  d["bbox"][2] + x0, d["bbox"][3] + y0]
                    yolo_dets_all.append(dd)

            res = fuse_and_decide(yolo_dets_all, blobs, h_inv)

        except (RuntimeError, cv2.error) as _align_err:
            # --- Safe Fallback: Full-Board YOLO (Branch A) -------------
            # Catches both alignment RuntimeError and OpenCV errors
            # (e.g. !dsize.empty() on degenerate frames) so live demos
            # never crash; unknown pinned IDs also land here with a note.
            if not allow_yolo_fallback:
                raise RuntimeError(str(_align_err)) from _align_err
            alignment_mode = "yolo_fallback"
            board_id = None
            inliers = None
            t_align = 0.0
            t_diff = 0.0
            timings["align"] = t_align
            timings["diff"] = t_diff
            t0 = time.perf_counter()
            try:
                # YOLODetector.predict expects RGB (internally converts to BGR),
                # while work_img is BGR -> convert before calling.
                yolo_dets = self.yolo.predict(
                    cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
            except Exception as e:
                # No inspection evidence exists: callers must handle an error,
                # not classify an empty detection list as a passing board.
                raise RuntimeError(
                    "Inspection unavailable: alignment and YOLO fallback failed. "
                    "No board verdict or robot action was produced."
                ) from e
            else:
                timings["roi_yolo"] = (time.perf_counter() - t0) * 1000.0
                items = [{
                    "class_id": d["class_id"], "class_name": d["class_name"],
                    "conf": float(d["conf"]), "source": "yolo_fallback",
                    "bbox": [float(v) for v in d["bbox"]],
                    "z_diff": None, "area": None,
                    "diff_box": None, "centroid_input": None}
                    for d in yolo_dets]
                res = decide(items)
                res["items"] = items
            h_inv = None
            diff_map = None
            blobs = []
            cents = np.zeros((0, 2), np.float32)

        # --- Step 4: Shift coordinates back to original frame (+ROI offset)
        ox, oy = offset
        items = list(res.get("items", []) if res else [])
        for it in items:
            bb = it.get("bbox")
            if bb is not None:
                it["bbox"] = [bb[0] + ox, bb[1] + oy, bb[2] + ox, bb[3] + oy]
            ci = it.get("centroid_input")
            if ci is not None:
                it["centroid_input"] = [ci[0] + ox, ci[1] + oy]
        cents_global = [(float(c[0]) + ox, float(c[1]) + oy) for c in cents]

        # --- Step 5: 4-level decision -> Robot action ------------------
        level = int(res.get("level", 4)) if res else 4
        conclusion = res.get("conclusion", "PASS") if res else "PASS"
        verdict = res.get("verdict", "NONE") if res else "NONE"
        reasons = list(res.get("reasons", [])) if res else []
        robot_action = dict(LEVEL_TO_ROBOT.get(level, LEVEL_TO_ROBOT[4]))

        timings["total"] = (time.perf_counter() - t_total) * 1000.0

        # Surface operational notes without breaking the result contract:
        # - unknown pinned Board ID (user typo) -> fallback ran instead.
        # - YOLO fallback after failed alignment.
        notes: list[str] = []
        if pinned_unknown:
            notes.append(
                f"Pinned Board ID '{pinned_board_id}' not found in cache "
                f"(boards {list(self.cache.get('board_ids', []))}); "
                "ran full matching / YOLO fallback instead.")
        if alignment_mode == "yolo_fallback":
            notes.append(
                "Board alignment failed (low ORB inliers); "
                "used safe full-board YOLO fallback.")
        reasons_out = (notes + reasons) if notes else reasons

        # --- Step 6: Visuals -------------------------------------------
        annotated_bgr = None
        heatmap_bgr = None
        if include_visuals:
            annotated_bgr = self._draw_annotated(img, items, cents_global,
                                                 roi_rect, offset,
                                                 work_img.shape[:2])
            heatmap_bgr = self._make_heatmap(diff_map, h_inv,
                                             (h_img, w_img), offset,
                                             work_img.shape[:2],
                                             fallback=(alignment_mode == "yolo_fallback"),
                                             base_bgr=img)

        return {
            "status": "ok",
            "board_id": board_id,
            "inliers": inliers,
            "alignment_mode": alignment_mode,
            "level": level,
            "conclusion": conclusion,
            "verdict": verdict,
            "reasons": reasons_out,
            "robot_action": robot_action,
            "items": items,
            "timings_ms": timings,
            "annotated_bgr": annotated_bgr,
            "heatmap_bgr": heatmap_bgr,
            "raw_bgr": img,
        }

    # ------------------------------------------------------------------ #
    # Helpers (private)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_bgr(image_bgr: np.ndarray) -> np.ndarray:
        # Strict gate: reject bad inputs loudly (ValueError) instead of
        # returning a silent PASS. Catches None, grayscale, float, BGRA,
        # empty and tiny frames (min side < 32px is unstable for ORB/RANSAC
        # and crashes cv2.resize with !dsize.empty()).
        if image_bgr is None:
            raise ValueError("image_bgr is None (no frame grabbed)")
        img = np.asarray(image_bgr)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"image_bgr must be an ndarray (H, W, 3), BGR uint8 "
                f"(got shape {getattr(img, 'shape', None)})")
        if img.dtype != np.uint8:
            raise ValueError(
                f"image_bgr dtype must be uint8 (got {img.dtype})")
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Empty input image (dimensions are zero)")
        if min(h, w) < 32:
            raise ValueError(
                f"Image too small for ORB/RANSAC (got {w}x{h}, "
                f"min side 32px required)")
        return img

    @staticmethod
    def _crop_roi(img: np.ndarray,
                  roi_rect: list[int]) -> tuple[np.ndarray, tuple[int, int]]:
        """Crop Fixed Jig ROI region. Returns (work_img, offset=(x, y)).

        roi_rect = [x, y, w, h]. If invalid (out of bounds / empty),
        returns (img, (0, 0)) without cropping.
        """
        if (not isinstance(roi_rect, (list, tuple)) or len(roi_rect) != 4):
            return img, (0, 0)
        x, y, w, h = [int(v) for v in roi_rect]
        H, W = img.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x >= W or y >= H:
            return img, (0, 0)
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        if x2 <= x or y2 <= y:
            return img, (0, 0)
        return img[y:y2, x:x2], (x, y)

    def _template_gray(self, board_id: str,
                       like_shape: tuple[int, int]) -> np.ndarray:
        cache_key = (str(board_id), (int(like_shape[0]), int(like_shape[1])))
        hit = self._tpl_gray_cache.get(cache_key)
        if hit is not None:
            return hit
        import src.reference.board_id_and_align as _ba  # noqa: E402
        p = _ba._tpl_image_path(board_id)
        if p is None:
            # Fallback: empty grayscale template (no data -> diff = 0).
            return np.zeros(like_shape, np.uint8)
        data = np.fromfile(str(p), dtype=np.uint8)
        tpl = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            return np.zeros(like_shape, np.uint8)
        if tpl.shape != like_shape:
            tpl = cv2.resize(tpl, (like_shape[1], like_shape[0]),
                             interpolation=cv2.INTER_AREA)
        self._tpl_gray_cache[cache_key] = tpl
        return tpl

    def _draw_annotated(self, img_bgr: np.ndarray, items: list[dict[str, Any]],
                        centroids: list[tuple[float, float]],
                        roi_rect: list[int] | None,
                        offset: tuple[int, int],
                        work_shape: tuple[int, int]) -> np.ndarray:
        vis = img_bgr.copy()
        ox, oy = offset
        if roi_rect is not None:
            x, y, w, h = [int(v) for v in roi_rect]
            cv2.rectangle(vis, (x, y), (min(x + w, vis.shape[1]),
                                        min(y + h, vis.shape[0])),
                          (255, 200, 0), 3)
            cv2.putText(vis, "JIG ROI", (x + 6, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
        for it in items:
            bb = it.get("bbox")
            if bb is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bb]
            cid = it.get("class_id")
            color = CLASS_COLORS_BGR.get(int(cid) if cid is not None else -1,
                                         (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            label = f"{it.get('class_name', '?')} {float(it['conf'] or 0):.2f} [{it.get('source', '?')}]"
            (tw_, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y1t = max(0, y1 - th_ - 8)
            cv2.rectangle(vis, (x1, y1t), (x1 + tw_ + 8, y1t + th_ + 8), color, -1)
            cv2.putText(vis, label, (x1 + 4, y1t + th_ + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        # Blob centroids (projected via H_inv + offset) -> white target rings.
        for (px, py) in centroids:
            x, y = int(round(px)), int(round(py))
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (x, y), 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        return vis

    def _make_heatmap(self, diff_map: np.ndarray | None,
                      h_inv: np.ndarray | None,
                      img_shape: tuple[int, int],
                      offset: tuple[int, int],
                      work_shape: tuple[int, int],
                      fallback: bool = False,
                      base_bgr: np.ndarray | None = None,
                      alpha: float = 0.45) -> np.ndarray:
        h_img, w_img = int(img_shape[0]), int(img_shape[1])
        if fallback or diff_map is None or h_inv is None:
            heat = np.zeros((h_img, w_img, 3), np.uint8)
            txt = "YOLO Fallback Mode - No Template Matched"
            (tw_, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cx, cy = w_img // 2, h_img // 2
            cv2.putText(heat, txt, (cx - tw_ // 2, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (180, 180, 180), 2, cv2.LINE_AA)
            return heat
        # Lock 1: warp diff_map (Template space) through H_inv -> work_img space.
        tw_, th_ = int(work_shape[1]), int(work_shape[0])
        diff_in_work = cv2.warpPerspective(
            np.asarray(diff_map, dtype=np.float32), np.asarray(h_inv, np.float64),
            (tw_, th_), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Shift by ROI offset back to full frame (pad outside ROI with zeros).
        canvas = np.zeros((h_img, w_img), np.float32)
        ox, oy = offset
        y2 = min(oy + th_, h_img)
        x2 = min(ox + tw_, w_img)
        if y2 > oy and x2 > ox:
            canvas[oy:y2, ox:x2] = diff_in_work[:y2 - oy, :x2 - ox]
        mx = float(np.max(canvas)) if canvas.size else 0.0
        if not np.isfinite(mx) or mx <= 1e-9:
            norm = np.zeros_like(canvas, dtype=np.uint8)
        else:
            norm = np.clip(canvas / mx * 255.0, 0, 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        if base_bgr is not None and base_bgr.shape[:2] == heat_bgr.shape[:2]:
            return cv2.addWeighted(base_bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)
        return heat_bgr

    # ------------------------------------------------------------------ #
    # Serialization helpers (used by AI Server / API)
    # ------------------------------------------------------------------ #
    @staticmethod
    def encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 85,
                        max_side: int | None = None) -> str:
        work = img_bgr
        if max_side is not None and max_side > 0:
            h, w = img_bgr.shape[:2]
            m = max(h, w)
            if m > max_side:
                scale = float(max_side) / float(m)
                work = cv2.resize(
                    img_bgr,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
        ok, buf = cv2.imencode(".jpg", work,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def to_api_dict(self, result: dict[str, Any],
                    include_images: bool = True,
                    preview_max_side: int | None = None) -> dict[str, Any]:
        """Convert inspection result into a JSON-serializable API dictionary."""
        out = {
            "status": result.get("status", "ok"),
            "board_id": result.get("board_id"),
            "inliers": result.get("inliers"),
            "alignment_mode": result.get("alignment_mode"),
            "level": result.get("level"),
            "conclusion": result.get("conclusion"),
            "verdict": result.get("verdict"),
            "reasons": result.get("reasons", []),
            "robot_action": result.get("robot_action"),
            "items": result.get("items", []),
            "timings_ms": result.get("timings_ms", {}),
        }
        if include_images:
            ab = result.get("annotated_bgr")
            hb = result.get("heatmap_bgr")
            rb = result.get("raw_bgr")
            q = 80 if preview_max_side else 85
            out["annotated_jpeg_b64"] = (
                self.encode_jpeg_b64(ab, quality=q, max_side=preview_max_side)
                if ab is not None else ""
            )
            out["heatmap_jpeg_b64"] = (
                self.encode_jpeg_b64(hb, quality=q, max_side=preview_max_side)
                if hb is not None else ""
            )
            out["raw_jpeg_b64"] = (
                self.encode_jpeg_b64(rb, quality=q, max_side=preview_max_side)
                if rb is not None else ""
            )
        return out
