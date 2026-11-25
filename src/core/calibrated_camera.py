from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.reference import tolerance_diff as td

TOP_K = 8
MIN_AREA = 100
MIN_Z = 3.5
ALIGN_SCALE = .65
CONFIRM_RADIUS = 35
RECAPTURE_RATIO = 1.5


@dataclass(frozen=True)
class Region:
    center: tuple[float, float]
    box: tuple[int, int, int, int]
    score: float
    area: int


@dataclass(frozen=True)
class CameraResult:
    action: str
    seed: float | None
    regions: tuple[Region, ...] = ()
    pending: tuple[Region, ...] | None = None


def confirm_regions(first, second):
    distances = sorted(
        (float(np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])), i, j)
        for i, a in enumerate(first) for j, b in enumerate(second)
    )
    first_used, second_used = set(), set()
    for distance, i, j in distances:
        if distance > CONFIRM_RADIUS:
            break
        if i not in first_used and j not in second_used:
            first_used.add(i)
            second_used.add(j)
    return tuple(region for j, region in enumerate(second) if j in second_used)


def decide(seed, threshold, regions, pending=None):
    if seed is None or not np.isfinite(seed) or seed > threshold * RECAPTURE_RATIO:
        return CameraResult("RECAPTURE", seed, pending=pending)
    if pending is not None:
        return CameraResult("INSPECT", seed, confirm_regions(pending, regions))
    if seed > threshold:
        return CameraResult("CONFIRM", seed, pending=regions)
    return CameraResult("INSPECT", seed, regions)


class CalibratedCameraInspection:
    """Camera workflow for the measured Type A reference and capture setup."""

    def __init__(self, results_dir: Path, *, profile_dir: Path | None = None):
        settings_dir = profile_dir or results_dir / "real_pcb_branch_b_quality_20261116"
        settings = json.loads((settings_dir / "settings.json").read_text(encoding="utf-8"))
        profile = profile_dir or results_dir / settings["template_run"]
        self.threshold = float(settings["threshold"])
        self.template = self._read(profile / "template_A.png")
        self.gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.mask = self._read(profile / "template_mask.png", cv2.IMREAD_GRAYSCALE) > 0
        provenance = self._read(profile / "template_provenance.png", cv2.IMREAD_GRAYSCALE)
        original = np.load(profile / "residual_tolerance.npy", allow_pickle=False)
        if not (self.gray.shape == self.mask.shape == provenance.shape == original.shape):
            raise ValueError("Type A template, mask and tolerance map must have the same dimensions.")
        # Keep the validated quality score on the original tolerance map.
        self.quality_tolerance = original
        self.detection_tolerance = np.minimum(original, np.where(provenance > 1, 4., np.inf)).astype(np.float32)
        self.bright = cv2.dilate((self.gray >= 180).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        self.sift = cv2.SIFT_create(nfeatures=6500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.points, self.descriptors = self._features(self.template)

    @staticmethod
    def _read(path, flags=cv2.IMREAD_COLOR):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
        if image is None:
            raise ValueError(f"Cannot read calibration image: {path}")
        return image

    def _features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=ALIGN_SCALE, fy=ALIGN_SCALE, interpolation=cv2.INTER_AREA)
        contrast = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(small)
        keys, descriptors = self.sift.detectAndCompute(contrast, None)
        return np.float32([key.pt for key in keys]) / ALIGN_SCALE, descriptors

    def _align(self, frame):
        points, descriptors = self._features(frame)
        if descriptors is None:
            return None
        pairs = self.matcher.knnMatch(descriptors, self.descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
        if len(good) < 35:
            return None
        source = np.float32([points[m.queryIdx] for m in good])
        target = np.float32([self.points[m.trainIdx] for m in good])
        H, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0, maxIters=5000, confidence=.999)
        if H is None:
            return None
        keep = inliers.ravel().astype(bool)
        if keep.sum() < max(35, .2 * len(good)):
            return None
        projected = cv2.perspectiveTransform(source[keep][None], H)[0]
        if np.median(np.linalg.norm(projected - target[keep], axis=1)) > 2.5:
            return None
        return H

    def inspect(self, frame, pending=None):
        result, aligned, _ = self.inspect_with_transform(frame, pending)
        return result, aligned

    def inspect_with_transform(self, frame, pending=None):
        """Return the exact input-to-template transform for reproducible evaluation."""
        H = self._align(frame)
        if H is None:
            return decide(None, self.threshold, (), pending), frame, None
        h, w = self.gray.shape
        warped = cv2.warpPerspective(frame, H, (w, h))
        valid = cv2.warpPerspective(np.ones(frame.shape[:2], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        mask = self.mask & (valid > 0)
        if np.count_nonzero(mask) < .98 * np.count_nonzero(self.mask):
            return decide(None, self.threshold, (), pending), warped, H

        diff = td.run_branch_b(warped, self.gray, valid_mask=mask, auto_calib_tol=False)["diff_map"]
        quality_map = diff / self.quality_tolerance
        mean, std = (float(v.item()) for v in cv2.meanStdDev(quality_map))
        seed = mean + td.RUN_SEED_K * max(std, td.FLOOR_SEED)
        if seed > self.threshold * RECAPTURE_RATIO:
            return decide(seed, self.threshold, (), pending), warped, H

        blobs = td.extract_blobs(diff / self.detection_tolerance, min_area=MIN_AREA,
                                 seed_k=td.RUN_SEED_K, top_k=10000)
        ranked = sorted((b for b in blobs if b["z_diff"] >= MIN_Z), key=self._rank, reverse=True)
        regions = tuple(Region(tuple(b["centroid_tpl"]), tuple(b["bbox_tpl"]), b["z_diff"], int(b["area"]))
                        for b in ranked[:TOP_K])
        return decide(seed, self.threshold, regions, pending), warped, H

    def _rank(self, blob):
        x, y = np.rint(blob["centroid_tpl"]).astype(int)
        area_factor = np.clip(blob["area"] / 250, .4, 4) ** .25
        return blob["z_diff"] * area_factor * (.8 if self.bright[y, x] else 1)

