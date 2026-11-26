from unittest.mock import Mock

import numpy as np
import pytest

from src.core import real_ab_pipeline as ab
from src.core.calibrated_camera import CameraResult, Region
from src.decision.logic import decide


REGION = {"center": (20, 20), "box": (15, 15, 25, 25), "score": 15, "area_template": 100}


def detection(class_id=1, name="short", conf=0.9):
    return {"bbox": [10, 10, 30, 30], "class_id": class_id,
            "class_name": name, "conf": conf, "source": "yolo"}


@pytest.mark.parametrize("action", ["RECAPTURE", "CONFIRM"])
def test_actionable_yolo_outranks_capture_action(action):
    items = [detection()]
    result = ab.combine(items, action, [REGION])
    assert result["conclusion"] == "FAIL"
    assert result["items"] == items
    assert result.get("robot_action") is None


@pytest.mark.parametrize("action", ["RECAPTURE", "CONFIRM"])
def test_capture_action_still_outranks_low_confidence_yolo(action):
    result = ab.combine([detection(conf=0.25)], action, [REGION])
    assert result["conclusion"] == action


def test_unmatched_region_triggers_review():
    result = ab.combine([], "INSPECT", [REGION])
    assert result["conclusion"] == "REVIEW"
    assert "vùng lạ nhánh B" in result["reasons"]
    assert result["b_regions_unmatched"] == [REGION]


def test_weak_unmatched_region_does_not_force_review():
    weak = {**REGION, "score": ab.B_REVIEW_SCORE - 0.1}
    result = ab.combine([], "INSPECT", [weak])
    assert result["conclusion"] == "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"
    assert result["b_regions_unmatched"] == [weak]


def test_region_without_score_counts_as_strong():
    region = {key: value for key, value in REGION.items() if key != "score"}
    assert ab.combine([], "INSPECT", [region])["conclusion"] == "REVIEW"


@pytest.mark.parametrize(("item", "conclusion"), [
    (detection(), "FAIL"), (detection(3, "spur"), "REWORK"),
    # 3-class model (real_3class.pt): names map to the same internal ids
    (detection(5, "missing_hole"), "FAIL"), (detection(0, "open_circuit"), "FAIL"),
])
def test_unmatched_region_does_not_override_a(item, conclusion):
    region = {**REGION, "center": (40, 40)}
    assert ab.combine([item], "INSPECT", [region])["conclusion"] == conclusion


def test_no_detections_does_not_report_pass():
    result = ab.combine([], "INSPECT", [])
    assert result["conclusion"] == "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"
    assert result["verdict"] != "PASS"
    assert result.get("robot_action") is None


def test_overlap_does_not_change_a_decision():
    items = [detection(conf=0.9)]
    result = ab.combine(items, "INSPECT", [{**REGION, "box": (0, 0, 32, 32)}])
    assert result["b_regions_unmatched"] == []
    assert result["verdict"] == decide(items)["verdict"]
    assert result["items"] == items


@pytest.mark.parametrize("item", [
    detection(conf=0.25),                      # FAIL class below 0.55
    detection(5, "missing_hole", conf=0.50),   # FAIL class below 0.55
    detection(3, "spur", conf=0.30),           # REWORK class below 0.35
])
@pytest.mark.parametrize("regions", [[], [{**REGION, "box": (0, 0, 32, 32)}]])
def test_low_confidence_a_box_is_review_never_pass(item, regions):
    result = ab.combine([item], "INSPECT", regions)
    assert result["conclusion"] == "REVIEW"
    assert result["level"] == 3
    assert "PASS" not in (result["conclusion"], result["verdict"])


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(ab, "YOLODetector", Mock())
    monkeypatch.setattr(ab, "CalibratedCameraInspection", Mock())
    return ab.RealABInspection(tmp_path / "mock.pt", tmp_path)


@pytest.mark.parametrize("branch", ["a", "b"])
def test_inspect_propagates_errors(pipeline, branch):
    pipeline.yolo.predict.return_value = []
    call = pipeline.yolo.predict if branch == "a" else pipeline.camera.inspect_with_transform
    call.side_effect = RuntimeError("inspection error")
    with pytest.raises(RuntimeError, match="inspection error"):
        pipeline.inspect(np.zeros((64, 64, 3), np.uint8))


@pytest.mark.parametrize("has_transform", [False, True])
def test_inspect_projects_only_with_homography(pipeline, has_transform):
    frame = np.zeros((64, 64, 3), np.uint8)
    template_region = Region((25, 35), (20, 30, 30, 40), 4, 100)
    transform = np.array([[1, 0, 5], [0, 1, 15], [0, 0, 1]], float)
    pending = (template_region,)
    pipeline.yolo.predict.return_value = []
    pipeline.camera.inspect_with_transform.return_value = (
        CameraResult("INSPECT", 1.0, (template_region,)), frame,
        transform if has_transform else None,
    )
    result = pipeline.inspect(frame, pending)
    assert pipeline.camera.inspect_with_transform.call_args.args[1] is pending
    assert pipeline.yolo.predict.call_args.kwargs["imgsz"] == 2592
    assert pipeline.yolo.predict.call_args.kwargs["conf"] == 0.20
    assert pipeline.yolo.predict.call_args.kwargs["iou"] == 0.7
    assert result["annotated_bgr"].shape == frame.shape
    if has_transform:
        region, = result["b_regions_unmatched"]
        assert region["center"] == [20, 20]
        assert region["box"] == [15, 15, 25, 25]
    else:
        assert result["b_regions_unmatched"] == []


def test_inspect_without_branch_b_skips_camera(pipeline):
    pipeline.yolo.predict.return_value = [detection(conf=0.25)]
    result = pipeline.inspect(np.zeros((64, 64, 3), np.uint8), pending=("old",), use_b=False)
    pipeline.camera.inspect_with_transform.assert_not_called()
    assert result["b_action"] == "SKIPPED"
    assert result["branch_b"] is False
    assert result["b_pending"] is None
    assert result["conclusion"] == "REVIEW"
