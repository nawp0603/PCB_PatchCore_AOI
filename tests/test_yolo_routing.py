from types import SimpleNamespace

from src.decision.sorting import sort_decision
from src.models.yolo_detector import YOLODetector


def test_checkpoint_raw_class_is_preserved_for_routing():
    detector = YOLODetector.__new__(YOLODetector)
    detector.model = SimpleNamespace(names={2: "open_circuit"})
    box = SimpleNamespace(cls=[2], conf=[0.9], xyxy=[[1, 2, 3, 4]])

    item, = detector._parse_one(SimpleNamespace(boxes=[box]))

    assert item["class_id"] == 0
    assert item["routing_class_id"] == 2
    assert sort_decision({"conclusion": "FAIL", "items": [{**item, "source": "yolo"}]}) == {
        "action": "pick", "bin": "C13"}
