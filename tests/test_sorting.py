import pytest

from src.decision.sorting import sort_decision


@pytest.mark.parametrize(("conclusion", "action", "bin_name"), [
    ("FAIL", "pick", "FAIL"),
    ("REWORK", "pick", "FAIL"),
    ("KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "pick", "PASS"),
    ("REVIEW", "operator", None),
    ("CONFIRM", "none", None),
    ("RECAPTURE", "none", None),
    ("UNAVAILABLE", "none", None),
    ("PASS", "none", None),        # unknown to the A+B pipeline -> never picked
    (None, "none", None),
])
def test_sort_decision(conclusion, action, bin_name):
    assert sort_decision({"conclusion": conclusion}) == {"action": action, "bin": bin_name}
