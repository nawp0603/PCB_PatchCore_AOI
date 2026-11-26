import pytest

from src.decision.sorting import sort_decision


@pytest.mark.parametrize(("conclusion", "action", "bin_name"), [
    ("KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "pick", "A11"),
    ("REVIEW", "none", None),
    ("CONFIRM", "none", None),
    ("RECAPTURE", "none", None),
    ("UNAVAILABLE", "none", None),
    ("PASS", "none", None),        # unknown to the A+B pipeline -> never picked
    (None, "none", None),
])
def test_sort_decision(conclusion, action, bin_name):
    assert sort_decision({"conclusion": conclusion}) == {"action": action, "bin": bin_name}


@pytest.mark.parametrize(("class_id", "bin_name"), [(0, "C11"), (1, "C12"), (2, "C13")])
def test_yolo_error_class_selects_physical_error_bin(class_id, bin_name):
    result = {"conclusion": "FAIL", "items": [{"source": "yolo", "class_id": class_id}]}
    assert sort_decision(result) == {"action": "pick", "bin": bin_name}


def test_unclassified_or_non_yolo_error_never_picks():
    assert sort_decision({"conclusion": "FAIL"}) == {"action": "none", "bin": None}
    assert sort_decision({"conclusion": "REWORK", "items": [
        {"source": "patchcore", "class_id": 2}]}) == {"action": "none", "bin": None}


def test_raw_checkpoint_class_routes_even_when_canonical_class_differs():
    result = {"conclusion": "FAIL", "items": [
        {"source": "yolo", "class_id": 0, "routing_class_id": 2}]}
    assert sort_decision(result) == {"action": "pick", "bin": "C13"}


def test_most_frequent_routable_raw_class_selects_error_bin():
    result = {"conclusion": "FAIL", "items": [
        {"source": "yolo", "class_id": 0, "routing_class_id": 1},
        {"source": "yolo", "class_id": 2, "routing_class_id": 2},
        {"source": "yolo", "class_id": 0, "routing_class_id": 1},
    ]}
    assert sort_decision(result) == {"action": "pick", "bin": "C12"}
