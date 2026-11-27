"""real_3class.pt class names must map onto the internal ids used by the decision rule."""
from pathlib import Path

import pytest

from src.decision.logic import FAIL_CLASSES, MEDIUM_FAIL_CLASSES
from src.models.yolo_detector import _NAME_TO_ID

WEIGHTS = Path(__file__).resolve().parents[1] / "assets" / "weights" / "pcb_camera" / "real_3class.pt"


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="real_3class.pt not present")
def test_3class_names_map_to_fail_classes():
    from ultralytics import YOLO
    names = YOLO(str(WEIGHTS)).names
    assert sorted(names.values()) == ["missing_hole", "open_circuit", "spur"]
    ids = {name: _NAME_TO_ID[name] for name in names.values()}
    assert ids["missing_hole"] in FAIL_CLASSES
    assert ids["open_circuit"] in FAIL_CLASSES
    assert ids["spur"] in MEDIUM_FAIL_CLASSES
