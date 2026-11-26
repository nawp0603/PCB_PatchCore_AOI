"""Map an inspection conclusion to a robot sorting action. No robot I/O here."""

CLEAN = "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"
PASS_BIN = "A11"
YOLO_ERROR_BINS = {0: "C11", 1: "C12", 2: "C13"}


def sort_decision(result: dict) -> dict:
    """Return an automatic pick only for a routable inspection conclusion."""
    conclusion = result.get("conclusion")
    if conclusion in ("FAIL", "REWORK"):
        counts = {class_id: 0 for class_id in YOLO_ERROR_BINS}
        for item in result.get("a_items", result.get("items", ())):
            if item.get("source") != "yolo":
                continue
            class_id = item.get("routing_class_id", item.get("class_id"))
            if class_id in counts:
                counts[class_id] += 1
        eligible = [class_id for class_id, count in counts.items() if count]
        if eligible:
            dominant = max(eligible, key=lambda class_id: (counts[class_id], -class_id))
            return {"action": "pick", "bin": YOLO_ERROR_BINS[dominant]}
        return {"action": "none", "bin": None}
    if conclusion == CLEAN:
        return {"action": "pick", "bin": PASS_BIN}
    return {"action": "none", "bin": None}
