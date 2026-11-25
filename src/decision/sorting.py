"""Map an inspection conclusion to a robot sorting action. No robot I/O here."""

CLEAN = "KHÔNG THẤY LỖI (chưa hiệu chuẩn)"
FAIL_BIN = "FAIL"
PASS_BIN = "PASS"


def sort_decision(result: dict) -> dict:
    """Return {"action": "pick" | "operator" | "none", "bin": "FAIL" | "PASS" | None}.

    pick     -> robot sorts automatically into "bin".
    operator -> the operator chooses the bin (REVIEW).
    none     -> no pick: capture again, confirm, error or unknown conclusion.
    """
    conclusion = result.get("conclusion")
    if conclusion in ("FAIL", "REWORK"):
        return {"action": "pick", "bin": FAIL_BIN}
    if conclusion == CLEAN:
        return {"action": "pick", "bin": PASS_BIN}
    if conclusion == "REVIEW":
        return {"action": "operator", "bin": None}
    return {"action": "none", "bin": None}
