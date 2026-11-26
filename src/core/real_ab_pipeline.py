"""Full-frame YOLO and calibrated Type A inspection; PatchCore for designs B, C, D, G."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.core.calibrated_camera import CalibratedCameraInspection
from src.decision.logic import decide
from src.models.yolo_detector import YOLODetector

# A Branch B region outside every A box only forces REVIEW at this score or above.
# Development measurement on the 40 design-A images with labelled defects masked out
# ("virtual clean boards"): the default threshold (~5.5) left 35/36 boards in REVIEW,
# score >= 12 left 4/36. Not yet checked on a physical GOOD board.
B_REVIEW_SCORE = 12.0

# Designs B, C, D, G: position-aware PatchCore (patchcore_inspector.py). A PatchCore region outside every
# YOLO box forces REVIEW at its design's P_REVIEW_SCORE (1.30, calibrated so that <=5% of held-out GOOD
# captures of B/C are flagged). PatchCore never gives FAIL by itself: the class and the FAIL bin stay
# with YOLO. If the board cannot be aligned to the reference, a YOLO FAIL/REWORK is kept, otherwise the
# board must be re-captured.
PATCHCORE_SOURCE = "patchcore"


def combine(a_items, b_action, b_regions_input, review_score=None, region_reason="vùng lạ nhánh B") -> dict:
    review_score = B_REVIEW_SCORE if review_score is None else review_score
    unmatched = []
    for region in b_regions_input:
        x, y = region["center"]
        if not any(l <= x <= r and t <= y <= b for l, t, r, b in
                   (item["bbox"] for item in a_items)):
            unmatched.append(region)

    a_decision = decide(a_items)
    if b_action in ("RECAPTURE", "CONFIRM") and a_decision["conclusion"] not in ("FAIL", "REWORK"):
        hint = ("Check PCB placement and capture again." if b_action == "RECAPTURE"
                else "Keep the same PCB and press Space for a second capture.")
        decision = dict(level=3, conclusion=b_action, verdict=b_action,
                        reasons=[hint], details=[])
    else:
        decision = a_decision
        if decision["conclusion"] not in ("FAIL", "REWORK"):
            # decide() only rescues low-confidence A boxes when Branch B supplies z_diff;
            # here any A box (conf >= operating threshold) must still prevent a clean result.
            reasons = []
            if a_items:
                reasons.append("A phát hiện lỗi độ tin cậy thấp (dưới ngưỡng FAIL/REWORK)")
            # A region without a score is treated as strong, so it can never pass silently.
            if any(region.get("score", float("inf")) >= review_score for region in unmatched):
                reasons.append(region_reason)
            if reasons:
                decision = {**decision, "level": 3, "conclusion": "REVIEW",
                            "verdict": "REVIEW", "reasons": reasons}
            else:
                decision = {**decision, "conclusion": "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"}
    return {**decision, "items": list(a_items), "b_regions_unmatched": unmatched}


def combine_patchcore(a_items, p_action, p_regions, review_score) -> dict:
    """YOLO + PatchCore. Unaligned board: keep a YOLO FAIL/REWORK, otherwise re-capture."""
    if p_action == "RECAPTURE":
        result = combine(a_items, "INSPECT", [])
        if result["conclusion"] in ("FAIL", "REWORK"):
            return result
        result = combine(a_items, "RECAPTURE", [])
        return {**result, "reasons": ["Không căn được bo với ảnh mẫu (PatchCore): đặt lại bo và chụp lại."]}
    return combine(a_items, "INSPECT", p_regions, review_score=review_score, region_reason="vùng lạ PatchCore")


def _project_regions(regions, homography):
    if homography is None or not regions:
        return []
    inverse = np.linalg.inv(homography)
    projected = []
    for region in regions:
        l, t, r, b = region.box
        points = np.float32([region.center, (l, t), (r, t), (r, b), (l, b)])
        center, *corners = cv2.perspectiveTransform(points[None], inverse)[0]
        left_top, right_bottom = np.min(corners, axis=0), np.max(corners, axis=0)
        projected.append(dict(center=center.tolist(),
                              box=np.concatenate([left_top, right_bottom]).tolist(),
                              score=float(region.score), area_template=float(region.area)))
    return projected


def _annotate(frame, items, regions, p_regions=()):
    overlay = frame.copy()
    boxes = [
        (item["bbox"], f"A: {item['class_name']} {item['conf']:.2f}", (0, 220, 70))
        for item in items
    ] + [(region["box"], "B: unusual region", (0, 165, 255)) for region in regions] + [
        (region["box"], f"P: anomaly {region['score']:.2f}", (247, 85, 168)) for region in p_regions]
    for box, label, color in boxes:
        x, y, r, b = (round(v) for v in box)
        cv2.rectangle(overlay, (x, y), (r, b), color, 3)
        origin = (max(x, 0), max(y - 6, 24))
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    color, 2, cv2.LINE_AA)
    return overlay


class RealABInspection:
    def __init__(self, weights_path, results_dir, profile_dir=None,
                 imgsz=2592, conf=0.20, iou=0.7, patchcore_dirs=None, backbone_weights=None):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.yolo = YOLODetector(weights_path, conf_threshold=conf, imgsz=imgsz)
        self.camera = CalibratedCameraInspection(Path(results_dir), profile_dir=profile_dir)
        self.patchcore_dirs = dict(patchcore_dirs or {})   # design -> (profile dir, memory dir)
        self.backbone_weights = backbone_weights
        self._backbone = None
        self._patchcore = (None, None)                       # (design, inspector); one design on the GPU

    def _patchcore_for(self, design):
        if self._patchcore[0] == design:
            return self._patchcore[1]
        from src.core.patchcore_branch import Backbone          # torch model, loaded on first use
        from src.core.patchcore_inspector import PatchCoreInspector
        if self._patchcore[1] is not None:
            self._patchcore[1].release()
        if self._backbone is None:
            self._backbone = Backbone(self.backbone_weights)
        profile, memory = self.patchcore_dirs[design]
        self._patchcore = (design, PatchCoreInspector(self._backbone, profile, memory))
        return self._patchcore[1]

    def _predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.yolo.predict(rgb, imgsz=self.imgsz, conf=self.conf,
                                 iou=self.iou, fallback_on_oom=False)

    def warmup(self) -> None:
        self._predict(np.zeros((1944, 2592, 3), np.uint8))

    def inspect(self, frame, pending=None, use_b=True, patchcore_design=None) -> dict:
        """use_b=False: board of another design, Branch B has no template for it.
        patchcore_design: with use_b=False, a design with a PatchCore memory (B, C, D, G)."""
        start = perf_counter()
        a_items = [{**item, "source": "yolo"} for item in self._predict(frame)]
        a_done = perf_counter()
        if use_b:
            b_result, _, homography = self.camera.inspect_with_transform(frame, pending)
            b_action, b_pending = b_result.action, b_result.pending
            b_regions = _project_regions(b_result.regions, homography)
        else:
            b_action, b_pending, b_regions = "INSPECT", None, []
        b_done = perf_counter()
        use_p = not use_b and patchcore_design in self.patchcore_dirs
        p_action, p_regions, p_review = "SKIPPED", [], None
        if use_p:
            inspector = self._patchcore_for(patchcore_design)
            p = inspector.inspect(frame)
            p_action, p_regions, p_review = p["action"], p["regions"], inspector.review_score
            result = combine_patchcore(a_items, p_action, p_regions, p_review)
            p_unmatched, unmatched = result["b_regions_unmatched"], []
        else:
            result = combine(a_items, b_action, b_regions)
            p_unmatched, unmatched = [], result["b_regions_unmatched"]
        p_done = perf_counter()
        annotated = _annotate(frame, a_items, unmatched, p_unmatched)
        b_items = [dict(class_name="Vùng lạ nhánh B", source="branchb", bbox=region["box"],
                        z_diff=region["score"], area=region["area_template"],
                        b_weak=region["score"] < B_REVIEW_SCORE) for region in unmatched]
        p_items = [dict(class_name="Vùng lạ PatchCore", source=PATCHCORE_SOURCE, bbox=region["box"],
                        z_diff=region["score"], area=region["area_template"],
                        p_weak=region["score"] < p_review) for region in p_unmatched]
        a_ms, b_ms = (a_done - start) * 1000, (b_done - a_done) * 1000
        return {**result, "items": [*result["items"], *b_items, *p_items], "b_regions_unmatched": unmatched,
                "patchcore_design": patchcore_design if use_p else None,
                "p_action": p_action, "p_regions": p_regions, "p_review_score": p_review,
                "p_ms": (p_done - b_done) * 1000,
                "a_items": a_items, "raw_bgr": frame, "annotated_bgr": annotated,
                "heatmap_bgr": None, "alignment_mode": "calibrated",
                "b_action": b_action if use_b else "SKIPPED", "b_pending": b_pending,
                "branch_b": use_b,
                "a_ms": a_ms, "b_ms": b_ms,
                "timings_ms": {"total": (perf_counter() - start) * 1000}}
