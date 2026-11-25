"""Branch R regions of designs B/C/D/G are shown to the operator only: they never change the
conclusion or the robot bin (offline test 24/11/2026, results/branch_r_bcdg_20261124)."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.core import real_ab_pipeline as ab
from src.core.calibrated_camera import CameraResult, Region
from src.decision.sorting import sort_decision
from src.ui import tkinter_app as ui

IDENTITY = np.eye(3)
STRONG = Region((32, 32), (24, 24, 40, 40), 99.0, 200)   # far above any review threshold


def detection(name="open_circuit", class_id=0, conf=0.9):
    return {"bbox": [2, 2, 10, 10], "class_id": class_id, "class_name": name, "conf": conf}


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(ab, "YOLODetector", Mock())
    monkeypatch.setattr(ab, "CalibratedCameraInspection", Mock())
    insp = ab.RealABInspection(tmp_path / "mock.pt", tmp_path, display_profiles={"D": tmp_path / "D"})
    insp.yolo.predict.return_value = []
    return insp


def run(pipeline, design, action="INSPECT"):
    camera = ab.CalibratedCameraInspection.return_value
    regions = (STRONG,) if action == "INSPECT" else ()
    camera.inspect_with_transform.return_value = (
        CameraResult(action, 1.0, regions, pending=None if action == "INSPECT" else (STRONG,)), None, IDENTITY)
    return pipeline.inspect(np.zeros((64, 64, 3), np.uint8), use_b=False, display_design=design)


@pytest.mark.parametrize("yolo", [[], [detection()], [detection("spur", 2, 0.5)], [detection(conf=0.3)]])
def test_view_regions_never_change_conclusion_or_bin(pipeline, yolo):
    pipeline.yolo.predict.return_value = yolo
    shown = run(pipeline, "D")
    pipeline.display_profiles = {}
    plain = run(pipeline, "D")
    assert shown["conclusion"] == plain["conclusion"]
    assert shown["verdict"] == plain["verdict"]
    assert sort_decision(shown) == sort_decision(plain)
    assert shown["b_regions_unmatched"] == [] and shown["b_action"] == "SKIPPED"
    view, = [item for item in shown["items"] if item["source"] == ab.DISPLAY_SOURCE]
    assert view["display_only"] is True and view["bbox"] == [24.0, 24.0, 40.0, 40.0]
    assert shown["display_design"] == "D" and plain["display_design"] is None


def test_clean_board_stays_not_review_with_strong_view_region(pipeline):
    result = run(pipeline, "D")
    assert result["conclusion"] == "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"
    assert len(result["r_regions"]) == 1


@pytest.mark.parametrize("action", ["CONFIRM", "RECAPTURE"])
def test_view_capture_state_does_not_ask_for_recapture(pipeline, action):
    result = run(pipeline, "D", action)
    assert result["conclusion"] == "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"
    assert result["r_action"] == action


def test_design_without_profile_or_design_a_skips_view(pipeline):
    camera = ab.CalibratedCameraInspection.return_value
    camera.inspect_with_transform.reset_mock()
    result = run(pipeline, "G")
    assert result["r_action"] == "SKIPPED" and result["r_regions"] == []
    camera.inspect_with_transform.assert_not_called()


def test_board_mode_helpers():
    mode = lambda value: SimpleNamespace(board_mode=SimpleNamespace(get=lambda: value))
    assert ui._use_branch_b(mode("A")) and ui._display_design(mode("A")) is None
    assert not ui._use_branch_b(mode("D")) and ui._display_design(mode("D")) == "D"
    assert not ui._use_branch_b(mode("OTHER")) and ui._display_design(mode("OTHER")) is None
    assert ui._branch_letter({"source": ab.DISPLAY_SOURCE}) == "R"
    assert "chỉ xem" in ui._item_label({"source": ab.DISPLAY_SOURCE})
    assert set(ui._display_profiles()) == {"B", "C", "D", "G"}
