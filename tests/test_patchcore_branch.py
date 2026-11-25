"""PatchCore branch (designs B, C, D, G): decision rules, positional distance, board-only alignment."""
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from src.core import real_ab_pipeline as ab
from src.core.board_align import BoardAligner, board_region
from src.core.patchcore_branch import _min_dist
from src.decision.sorting import sort_decision
from src.ui import tkinter_app as ui

T = 1.30
FAR = {"center": (60, 60), "box": (55, 55, 65, 65), "score": 2.0, "area_template": 100.0}


def det(name="open_circuit", cls=0, conf=0.9):
    return {"bbox": [0, 0, 20, 20], "class_id": cls, "class_name": name, "conf": conf, "source": "yolo"}


def test_strong_region_outside_yolo_is_review_never_fail():
    r = ab.combine_patchcore([], "INSPECT", [FAR], T)
    assert r["conclusion"] == "REVIEW" and "vùng lạ PatchCore" in r["reasons"]
    assert sort_decision(r)["action"] == "operator"


def test_weak_region_does_not_force_review():
    r = ab.combine_patchcore([], "INSPECT", [{**FAR, "score": T - 0.01}], T)
    assert r["conclusion"] == "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"


def test_yolo_fail_is_kept_with_patchcore_regions():
    r = ab.combine_patchcore([det()], "INSPECT", [FAR], T)
    assert r["conclusion"] == "FAIL" and sort_decision(r)["bin"] == "FAIL"


@pytest.mark.parametrize(("items", "expected"), [([det()], "FAIL"), ([det("spur", 2)], "REWORK"),
                                                  ([], "RECAPTURE"), ([det(conf=0.3)], "RECAPTURE")])
def test_unaligned_board_keeps_yolo_fail_else_recapture(items, expected):
    r = ab.combine_patchcore(items, "RECAPTURE", [], T)
    assert r["conclusion"] == expected


def test_branch_b_threshold_unchanged_for_design_a():
    assert ab.combine([], "INSPECT", [{**FAR, "score": 11.9}])["conclusion"] != "REVIEW"
    assert ab.combine([], "INSPECT", [{**FAR, "score": 12.0}])["conclusion"] == "REVIEW"


def test_min_dist_is_position_aware():
    torch.manual_seed(0)
    mem = torch.randn(1, 8, 12, 12)
    query = torch.roll(mem[0], shifts=1, dims=2)          # same content, shifted by one cell
    sq = mem.pow(2).sum(1)
    d1 = _min_dist(query, mem, sq, radius=1)[2:-2, 2:-2]
    d0 = _min_dist(query, mem, sq, radius=0)[2:-2, 2:-2]
    assert float(d1.max()) < 1e-3 < float(d0.min())


def _board(seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((600, 800, 3), np.uint8)
    img[100:500, 150:650] = (40, 120, 40)                  # green board on black background
    for _ in range(300):
        x, y = rng.integers(170, 630), rng.integers(120, 480)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 9)), tuple(int(v) for v in rng.integers(120, 255, 3)), -1)
    return img


def test_board_aligner_recovers_translation_and_checks_overlap():
    ref = _board()
    moved = cv2.warpAffine(ref, np.float32([[1, 0, 30], [0, 1, 20]]), (800, 600))
    H = BoardAligner(ref).align(moved)
    assert H is not None
    assert abs(H[0, 2] + 30) < 2 and abs(H[1, 2] + 20) < 2
    warped = board_region(cv2.warpPerspective(moved, H, (800, 600)))
    ref_board = board_region(ref)
    assert (warped & ref_board).sum() / (warped | ref_board).sum() > 0.95


def test_pipeline_uses_patchcore_only_for_its_designs(monkeypatch, tmp_path):
    monkeypatch.setattr(ab, "YOLODetector", Mock())
    monkeypatch.setattr(ab, "CalibratedCameraInspection", Mock())
    insp = ab.RealABInspection(tmp_path / "m.pt", tmp_path, patchcore_dirs={"D": ("p", "m")})
    insp.yolo.predict.return_value = []
    fake = SimpleNamespace(review_score=T, inspect=Mock(return_value=dict(action="INSPECT", regions=[FAR], max_score=2.0)))
    monkeypatch.setattr(insp, "_patchcore_for", lambda design: fake)
    frame = np.zeros((80, 80, 3), np.uint8)
    res = insp.inspect(frame, use_b=False, patchcore_design="D")
    assert res["conclusion"] == "REVIEW" and res["patchcore_design"] == "D"
    item, = [i for i in res["items"] if i["source"] == ab.PATCHCORE_SOURCE]
    assert item["p_weak"] is False
    plain = insp.inspect(frame, use_b=False, patchcore_design="G")      # no memory for G here
    assert plain["patchcore_design"] is None and plain["conclusion"] == "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"
    fake.inspect.assert_called_once()


def test_ui_helpers():
    mode = lambda v: SimpleNamespace(board_mode=SimpleNamespace(get=lambda: v))
    assert ui._use_branch_b(mode("A")) and ui._patchcore_design(mode("A")) is None
    assert not ui._use_branch_b(mode("C")) and ui._patchcore_design(mode("C")) == "C"
    assert ui._patchcore_design(mode("OTHER")) is None
    assert ui._branch_letter({"source": "patchcore"}) == "P"
    assert "PatchCore" in ui._board_text("D")
    assert set(ui._patchcore_designs()) == {"B", "C", "D", "G"}
