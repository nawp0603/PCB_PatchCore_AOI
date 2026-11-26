"""Branch A: Semantic Defect Detector (YOLO11n Full-Res).

Data contract:
  * predict(image): accepts RGB uint8 (H, W, 3) ndarray OR image path.
   * returns list[dict]: {"bbox": [x1, y1, x2, y2], "class_id": int,
     "routing_class_id": int, "class_name": str, "conf": float}, sorted by
     descending confidence.
    For batch input via predict_batch(): returns list[list[dict]].

Key invariants:
  * Full-res: imgsz=3072, conf=0.20.
  * Anti-Square-Padding: enforces rect=True, half=True on CUDA in every
    model.predict() call (reduces latency from 361ms to ~160-190ms).
  * predict_batch(): batches TOP_K 640x640 crops into a single GPU call,
    paired with spatial_center_gating(R=80px) to reject crop-border FPs.

Default weights: assets/weights/yolo11n_day2.pt (fine-tuned) or yolo11n_base.pt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = {
    0: "open_circuit",
    1: "short",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "missing_hole",
}
EXPECTED_NAMES = CLASS_NAMES
_NAME_TO_ID = {name: cid for cid, name in CLASS_NAMES.items()}

DEFAULT_IMGSZ = 3072
DEFAULT_CONF = 0.20
CENTER_GATE_RADIUS = 80.0  # px (plan.md 2.3 step 5)

# YOLO11s on MV-GE500C frames (<=2592x1944): cap at 2560 so we never upscale
# uselessly to 3072. Only applies when caller does NOT pass imgsz explicitly.
YOLO11S_IMGSZ_CAP = 2560

# --- PORTABLE hardware adaptation (Step 2, release engineer) ---
# Target machine may have low VRAM (4GB) or CPU-only. Gracefully downscale instead of crashing.
CPU_FALLBACK_IMGSZ = 1920   # CPU: cap resolution to prevent OOM and excessive latency
LOW_VRAM_GB = 6.0           # below this threshold: graceful downscale
LOW_VRAM_IMGSZ = 2560       # VRAM 4-6GB: cap at 2560 (keep rect+half)
TINY_VRAM_GB = 4.0          # below 4GB: cap at 1920
TINY_VRAM_IMGSZ = 1920
_OOM_FALLBACKS = (2560, 1920, 1280, 960)  # retry sequence on OutOfMemoryError


def _cuda_total_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
    except Exception:
        return 0.0


def resolve_device_and_imgsz(requested_device: str | None = None,
                             requested_imgsz: int | None = None
                             ) -> tuple[str, bool, int, str]:
    """Select (device, half, imgsz, reason) suited to available hardware.

    - No CUDA -> ("cpu", False, min(requested, 1920)).
      (half=True on CPU raises Half precision not supported.)
    - CUDA VRAM < 6GB -> cap 2560; < 4GB -> cap 1920 (keep half=True).
    - Otherwise preserve requested settings.
    - Never raises: any torch/CUDA query failure degrades to CPU FP32 1920
      so a station with a different GPU (discrete / integrated / none)
      stays ONLINE instead of going OFFLINE.
    """
    try:
        import torch
    except Exception as exc:
        want = int(requested_imgsz) if requested_imgsz else DEFAULT_IMGSZ
        want = min(want, CPU_FALLBACK_IMGSZ)
        return ("cpu", False, want,
                f"torch import failed ({exc}) -> CPU fallback, half=False")
    try:
        want_imgsz = int(requested_imgsz) if requested_imgsz else DEFAULT_IMGSZ
        if requested_device is None:
            try:
                has_cuda = bool(torch.cuda.is_available())
            except Exception:
                has_cuda = False
            device = "cuda" if has_cuda else "cpu"
        else:
            device = str(requested_device).lower()
            if device not in ("cuda", "cpu"):
                raise ValueError("device must be 'cuda', 'cpu' or None (auto)")
            if device == "cuda":
                try:
                    if not torch.cuda.is_available():
                        device = "cpu"
                except Exception:
                    device = "cpu"
        if device == "cpu":
            if want_imgsz > CPU_FALLBACK_IMGSZ:
                return (device, False, CPU_FALLBACK_IMGSZ,
                        f"CPU fallback: imgsz {want_imgsz}->{CPU_FALLBACK_IMGSZ}, half=False")
            return (device, False, want_imgsz, "CPU, half=False")
        # CUDA available
        try:
            total_gb = _cuda_total_gb()
        except Exception:
            total_gb = 0.0
        if total_gb > 0.0 and total_gb < TINY_VRAM_GB and want_imgsz > TINY_VRAM_IMGSZ:
            return (device, True, TINY_VRAM_IMGSZ,
                    f"VRAM {total_gb:.1f}GB<4GB: imgsz {want_imgsz}->{TINY_VRAM_IMGSZ}")
        if total_gb > 0.0 and total_gb < LOW_VRAM_GB and want_imgsz > LOW_VRAM_IMGSZ:
            return (device, True, LOW_VRAM_IMGSZ,
                    f"VRAM {total_gb:.1f}GB<6GB: imgsz {want_imgsz}->{LOW_VRAM_IMGSZ}")
        return (device, True, want_imgsz, f"GPU {total_gb:.1f}GB, keep imgsz={want_imgsz}")
    except ValueError:
        raise
    except Exception as exc:
        want = int(requested_imgsz) if requested_imgsz else CPU_FALLBACK_IMGSZ
        want = min(want, CPU_FALLBACK_IMGSZ)
        return ("cpu", False, want,
                f"device resolve failed ({exc}) -> CPU fallback, half=False")


class YOLODetector:
    def __init__(self, model_path: str | Path, conf_threshold: float = DEFAULT_CONF,
                 imgsz: int | None = None, device: str | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.conf_threshold = float(conf_threshold)
        # YOLO11s on MV-GE500C frames (<=2592x1944): cap at 2560 so we never
        # upscale uselessly to 3072. Only when caller did NOT pass imgsz
        # explicitly; DEFAULT_IMGSZ=3072 stays intact at module level (Lock 2).
        if imgsz is None:
            if "yolo11s" in self.model_path.stem.lower():
                imsz_input = YOLO11S_IMGSZ_CAP
            else:
                imsz_input = DEFAULT_IMGSZ
        else:
            # Ton tri nguyen tac override tu nguoi dung (vi du imgsz=3072).
            imsz_input = int(imgsz)
        self.requested_imgsz = int(imgsz) if imgsz is not None else imsz_input
        # PORTABLE: auto-adapt to hardware (CPU -> half=False + cap 1920;
        # VRAM <6GB -> 2560; <4GB -> 1920). Keep rect=True + half on CUDA.
        self.device, self.half, self.imgsz, self.adapt_note = \
            resolve_device_and_imgsz(device, imsz_input)
        if self.adapt_note and ("fallback" in self.adapt_note
                                or "VRAM" in self.adapt_note):
            print(f"[YOLODetector] {self.adapt_note}")
        self.model = YOLO(str(self.model_path), task="detect")

    # -- input preprocessing -----------------------------------------------
    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert RGB uint8 (contract) -> BGR for Ultralytics."""
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a np.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")
        if image.dtype != np.uint8:
            raise ValueError("image dtype must be uint8")
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _oom_retry(self, source, conf: float, imgsz: int, **nms):
        """Retry at 2560/1920 on OutOfMemoryError (keeping rect=True, half=True)."""
        import torch
        torch.cuda.empty_cache()
        last: Exception | None = None
        for fb in _OOM_FALLBACKS:
            if fb >= int(imgsz):
                continue
            try:
                print(f"[YOLODetector] OOM at imgsz={imgsz}, retrying {fb} "
                      f"(rect=True, half=True).")
                return self.model.predict(
                    source=source, conf=float(conf), imgsz=int(fb),
                    device=self.device, verbose=False,
                    rect=True, half=self.half,
                    quantize=(16 if self.half else None), **nms)
            except torch.cuda.OutOfMemoryError as e:
                last = e
                torch.cuda.empty_cache()
                continue
        # Final fallback: switch to CPU FP32 so 4GB GPUs never crash the app.
        try:
            print("[YOLODetector] CUDA OOM at all resolutions -> falling back to CPU (FP32).")
            self.device = "cpu"
            self.half = False
            self.imgsz = min(int(self.imgsz), CPU_FALLBACK_IMGSZ)
            self.adapt_note = f"{self.adapt_note} [CUDA OOM -> CPU FP32]".strip()
            return self.model.predict(
                source=source, conf=float(conf),
                imgsz=min(int(imgsz), CPU_FALLBACK_IMGSZ),
                device="cpu", verbose=False, rect=True, half=False,
                quantize=None, **nms)
        except Exception:
            pass
        raise last if last is not None else RuntimeError("OOM retry failed")

    # -- single image ------------------------------------------------------
    def predict(self, image: np.ndarray | str | Path,
                conf: float | None = None,
                imgsz: int | None = None, *,
                iou: float | None = None,
                fallback_on_oom: bool = True) -> list[dict[str, Any]]:
        """Full-board single-image inference. Enforces rect=True, half=self.half (Lock 3)."""
        if isinstance(image, (str, Path)):
            data = np.fromfile(str(image), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Cannot decode image: {image}")
        else:
            bgr = self._to_bgr(image)
        nms = {} if iou is None else {"iou": float(iou)}
        import torch
        try:
            # Lock 3: prevent square padding + use FP16 on CUDA in a single call.
            # NOTE: keep `half=` (required by tests) alongside `quantize=`
            # so ultralytics uses the unified scheme without deprecation spam.
            results = self.model.predict(
                source=bgr,
                conf=self.conf_threshold if conf is None else float(conf),
                imgsz=self.imgsz if imgsz is None else int(imgsz),
                device=self.device, verbose=False,
                rect=True, half=self.half,
                quantize=(16 if self.half else None), **nms)
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda" or not fallback_on_oom:
                raise
            results = self._oom_retry(
                bgr, self.conf_threshold if conf is None else float(conf),
                self.imgsz if imgsz is None else int(imgsz), **nms)
        return self._parse_results(results)

    # -- batch crops (Branch B ROI verification, Lock 3) --------------------
    def predict_batch(self, images: list[np.ndarray],
                      conf: float | None = None,
                      imgsz: int = 640) -> list[list[dict[str, Any]]]:
        """Single-batch GPU inference across N 640x640 crops (no per-crop loop)."""
        if not images:
            return []
        bgrs = [self._to_bgr(im) for im in images]
        import torch
        try:
            # Lock 3: single batched call, rect=True, half=self.half.
            # NOTE: keep `half=` (required by tests) alongside `quantize=`.
            results = self.model.predict(
                source=bgrs,
                conf=self.conf_threshold if conf is None else float(conf),
                imgsz=int(imgsz), device=self.device, verbose=False,
                rect=True, half=self.half,
                quantize=(16 if self.half else None))
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda":
                raise
            results = self._oom_retry(
                bgrs, self.conf_threshold if conf is None else float(conf),
                int(imgsz))
        return [self._parse_one(r) for r in results]

    # -- spatial center gating (Branch B step 5) -----------------------------
    @staticmethod
    def _box_iou(a: list[float], b: list[float]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0.0 else 0.0

    @staticmethod
    def spatial_center_gating(dets: list[dict[str, Any]],
                              center_xy: tuple[float, float],
                              radius: float = CENTER_GATE_RADIUS,
                              crop_origin: tuple[float, float] = (0.0, 0.0),
                              require_overlap: bool = True,
                              crop_size: int = 640,
                              diff_box: list[float] | None = None) -> list[dict[str, Any]]:
        """Retain box if its center lies within R=80px of diff centroid OR IoU > 0.

        dets: bbox in crop coordinates; center_xy/crop_origin in original image
        coordinates. Shifts box center to original image coordinates before comparison.

        require_overlap=True (default): also accepts boxes whose center lies outside R
        if they overlap the diff window (IoU > 0 with 2R square around diff centroid
        or provided diff_box).
        """
        cx, cy = float(center_xy[0]), float(center_xy[1])
        ox, oy = float(crop_origin[0]), float(crop_origin[1])
        r = float(radius)
        if diff_box is None:
            diff_win = [cx - r, cy - r, cx + r, cy + r]
        else:
            diff_win = [float(v) for v in diff_box]
        kept: list[dict[str, Any]] = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            gx1, gy1, gx2, gy2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
            bx = (gx1 + gx2) / 2.0
            by = (gy1 + gy2) / 2.0
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            if dist <= r:
                kept.append(d)
                continue
            if require_overlap and YOLODetector._box_iou(
                    [gx1, gy1, gx2, gy2], diff_win) > 0.0:
                kept.append(d)
        return kept

    # -- parse ---------------------------------------------------------------
    def _parse_one(self, r: Any) -> list[dict[str, Any]]:
        dets: list[dict[str, Any]] = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            # Imported checkpoints may order classes differently (e.g. PKU
            # alphabetical): trust the checkpoint's own name, then map it to the
            # repo's canonical ID so fusion/decision (keyed by ID) stay correct.
            name = str(self.model.names.get(cls_id, cls_id))
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": _NAME_TO_ID.get(name, cls_id),
                "routing_class_id": cls_id,
                "class_name": name,
                "conf": conf,
            })
        dets.sort(key=lambda d: d["conf"], reverse=True)
        return dets

    def _parse_results(self, results: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in results:
            out.extend(self._parse_one(r))
        out.sort(key=lambda d: d["conf"], reverse=True)
        return out
