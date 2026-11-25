"""SIFT-RANSAC alignment restricted to the board, with an independent overlap check.

Why: the capture background is a textured black surface that does not move between captures. With CLAHE
it gives many SIFT keypoints, and RANSAC can lock onto it and return ~identity while the board itself
moved (audit 24/11/2026: board-overlap IoU < 0.95 for 62/176 B, 112/115 C, 89/90 D, 27/40 G captures).

  1. keypoints only inside the green board region of the reference and of the capture
  2. Lowe ratio test (0.72); if that leaves too few pairs (rows of identical holes on design D make
     descriptors ambiguous), mutual nearest neighbours instead and let RANSAC sort them out
  3. accept H only if the warped board region overlaps the reference board region with IoU >= 0.95
"""
from __future__ import annotations

import cv2
import numpy as np

SCALE = 0.65
RATIO = 0.72
MIN_INLIERS = 35
MIN_BOARD_IOU = 0.95


def board_region(bgr: np.ndarray, dilate: int = 1) -> np.ndarray:
    """Convex hull of green solder-mask pixels (largest blob), optionally dilated."""
    b, g, r = cv2.split(bgr.astype(np.int16))
    mask = ((g > 25) & (g > r + 6) & (g > b + 4)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((65, 65), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.fillConvexPoly(out, cv2.convexHull(max(contours, key=cv2.contourArea)), 1)
    if dilate > 1:
        out = cv2.dilate(out, np.ones((dilate, dilate), np.uint8))
    return out > 0


MIN_ECC = 0.5


def _iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def _rect(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.minAreaRect(max(contours, key=cv2.contourArea)) if contours else None


class BoardAligner:
    def __init__(self, reference_bgr: np.ndarray):
        self.sift = cv2.SIFT_create(nfeatures=6500)
        self.ratio = cv2.BFMatcher(cv2.NORM_L2)
        self.mutual = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.shape = reference_bgr.shape[:2]
        self._ref_bgr = reference_bgr
        self.board = board_region(reference_bgr)
        self.points, self.descriptors = self._features(reference_bgr, self.board)
        self.last = {}

    def _features(self, bgr, board):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        small = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(small)
        region = cv2.dilate(board.astype(np.uint8), np.ones((41, 41), np.uint8))
        mask = cv2.resize(region, small.shape[::-1], interpolation=cv2.INTER_NEAREST)
        keys, desc = self.sift.detectAndCompute(small, mask)
        return np.float32([k.pt for k in keys]) / SCALE, desc

    def _homography(self, points, matches, min_ratio):
        if len(matches) < MIN_INLIERS:
            return None, 0, float("inf")
        src = np.float32([points[m.queryIdx] for m in matches])
        dst = np.float32([self.points[m.trainIdx] for m in matches])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=10000, confidence=.999)
        if H is None:
            return None, 0, float("inf")
        keep = inl.ravel().astype(bool)
        err = float(np.median(np.linalg.norm(cv2.perspectiveTransform(src[keep][None], H)[0] - dst[keep], axis=1)))
        if keep.sum() < max(MIN_INLIERS, min_ratio * len(matches)) or err > 2.5:
            return None, int(keep.sum()), err
        return H, int(keep.sum()), err

    def align(self, bgr: np.ndarray):
        """-> homography capture -> reference, or None (capture should be retaken)."""
        board = board_region(bgr)
        points, desc = self._features(bgr, board)
        self.last = dict(method=None, inliers=0, reproj=float("inf"), board_iou=0.0)
        if desc is None or self.descriptors is None or len(desc) < MIN_INLIERS:
            return None
        pairs = self.ratio.knnMatch(desc, self.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        candidates = [("ratio", good, 0.2), ("mutual", None, 0.0)]
        for method, matches, min_ratio in candidates:
            if matches is None:
                matches = self.mutual.match(desc, self.descriptors)
            H, n, err = self._homography(points, matches, min_ratio)
            if H is None:
                continue
            iou = self._board_iou(board, H)
            self.last = dict(method=method, inliers=n, reproj=err, board_iou=iou)
            if iou >= MIN_BOARD_IOU:
                return H
        H, cc = self._outline_ecc(bgr, board)
        if H is not None:
            iou = self._board_iou(board, H)
            self.last = dict(method="outline+ecc", inliers=0, reproj=float("nan"), board_iou=iou, ecc=cc)
            if iou >= MIN_BOARD_IOU and cc >= MIN_ECC:
                return H
        return None

    def _board_iou(self, board, H):
        warped = cv2.warpPerspective(board.astype(np.uint8), H, self.shape[::-1], flags=cv2.INTER_NEAREST) > 0
        return _iou(warped, self.board)

    # Fallback for boards with few distinctive keypoints: coarse pose from the board outline (4 rotations),
    # refined by ECC (Euclidean) on gradient images at 1/4 then 1/2 scale.
    def _grad(self, bgr, scale):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
        return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    def _outline_ecc(self, bgr, board):
        if not hasattr(self, "_ref_rect"):
            self._ref_rect = _rect(self.board)
            self._ref_bgr_grad = {s: self._grad(self._ref_bgr, s) for s in (0.25, 0.5)}
        rq, rr = _rect(board), self._ref_rect
        if rq is None or rr is None:
            return None, 0.0
        best = (None, -1.0)
        grads = {s: self._grad(bgr, s) for s in (0.25, 0.5)}
        for k, sign in [(k, s) for k in range(4) for s in (1, -1)]:
            A = cv2.getRotationMatrix2D(rq[0], sign * (rr[2] - rq[2]) + 90 * k, 1.0)
            A[:, 2] += np.array(rr[0]) - np.array(rq[0])
            warp, cc = A.astype(np.float32), -1.0
            try:
                for s in (0.25, 0.5):
                    w = warp.copy(); w[:, 2] *= s
                    mask = cv2.resize(self.board.astype(np.uint8), self._ref_bgr_grad[s].shape[::-1],
                                      interpolation=cv2.INTER_NEAREST)
                    cc, w = cv2.findTransformECC(self._ref_bgr_grad[s], grads[s], cv2.invertAffineTransform(w),
                                                 cv2.MOTION_EUCLIDEAN,
                                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), mask, 5)
                    warp = cv2.invertAffineTransform(w); warp[:, 2] /= s
            except cv2.error:
                continue
            if cc > best[1]:
                best = (np.vstack([warp, [0, 0, 1]]).astype(np.float64), float(cc))
        return best
