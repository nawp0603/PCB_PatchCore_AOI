"""Three-conclusion decision rule: FAIL / REVIEW / PASS (plan.md Section 2.4).

Each evidence item (Box Fusion / ensemble output) is a dict with fields:
  class_id (int), class_name (str), conf (float | None),   # S_final (0..1)
  source ("yolo"|"branchb"|"fused"), bbox (list[float]),
  z_diff (float | None), area (float | None).

Priority: FAIL > REVIEW > PASS.
No OR-style rule such as "anomaly_z >= 2.0 AND yolo_conf_max >= 0.30"
(if YOLO misses, the system would still pick PASS). The two REVIEW cases are
handled independently per rule table 2.4 (3a recall rescue for 6 classes;
3b open-set structural anomaly).
"""

from __future__ import annotations

from typing import Any

# --- Class groups per plan.md Section 2.4 ----------------------------------
FAIL_CLASSES = {0, 1, 5}        # open_circuit, short, missing_hole
MEDIUM_FAIL_CLASSES = {2, 3, 4}  # mouse_bite, spur, spurious_copper
ALL_CLASSES = FAIL_CLASSES | MEDIUM_FAIL_CLASSES

# --- Thresholds per plan.md Section 2.4 ------------------------------------
CONF_HIGH = 0.55        # level 1: S_final >= 0.55 -> FAIL / HIGH
CONF_MEDIUM = 0.35      # level 2: S_final >= 0.35 -> FAIL / MEDIUM
CONF_LOW_LO = 0.20      # level 3a: 0.20 <= S_final < 0.35 (low prediction)
Z_REVIEW = 2.0          # level 3a: Z_diff >= 2.0 in Branch B -> recall rescue
Z_OPENSET = 3.5         # level 3b: Z_diff >= 3.5 and Area >= 100 -> open-set
AREA_OPENSET = 100.0    # px^2, condition combined with Z_OPENSET


def decide(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert fused evidence items into a three-conclusion inspection decision.

    Returns:
      {"level": int, "conclusion": str, "verdict": str,
       "reasons": list[str], "details": list[dict]}
    """
    reasons: list[str] = []
    details: list[dict[str, Any]] = []

    def note(level: int, conclusion: str, verdict: str, why: str,
             item: dict[str, Any] | None = None) -> None:
        reasons.append(why)
        if item is not None:
            details.append({
                "level": level, "conclusion": conclusion, "verdict": verdict,
                "why": why, "class_id": item.get("class_id"),
                "class_name": item.get("class_name"), "conf": item.get("conf"),
                "source": item.get("source"), "z_diff": item.get("z_diff"),
                "area": item.get("area"), "bbox": item.get("bbox")})

    def _has(pred) -> list[dict[str, Any]]:
        return [it for it in items if pred(it)]

    # --- Level 1: FAIL / HIGH ----------------------------------------------
    fail_items = _has(lambda it: (
        it.get("class_id") in FAIL_CLASSES
        and it.get("conf") is not None
        and it["conf"] >= CONF_HIGH))
    if fail_items:
        for it in fail_items:
            note(1, "FAIL", "HIGH",
                 f"Critical defect '{it['class_name']}' (S_final={it['conf']:.3f} >= {CONF_HIGH})", it)
        return _pack(1, "FAIL", "HIGH", reasons, details)

    # --- Level 2: FAIL / MEDIUM --------------------------------------------
    medium_fail_items = _has(lambda it: (
        it.get("class_id") in MEDIUM_FAIL_CLASSES
        and it.get("conf") is not None
        and it["conf"] >= CONF_MEDIUM))
    if medium_fail_items:
        for it in medium_fail_items:
            note(2, "FAIL", "MEDIUM",
                 f"Defect '{it['class_name']}' (S_final={it['conf']:.3f} >= {CONF_MEDIUM})", it)
        return _pack(2, "FAIL", "MEDIUM", reasons, details)

    # --- Level 3a: REVIEW recall rescue (6 classes, low conf, Branch B Z>=2.0)
    review3a = _has(lambda it: (
        it.get("class_id") in ALL_CLASSES
        and it.get("conf") is not None
        and CONF_LOW_LO <= it["conf"] < CONF_MEDIUM
        and it.get("z_diff") is not None
        and it["z_diff"] >= Z_REVIEW))
    if review3a:
        # Escalate low-confidence detection corroborated by Branch B to REVIEW.
        for it in review3a:
            note(3, "REVIEW", "RECALL_RESCUE",
                 f"Low-confidence '{it['class_name']}' (S_final={it['conf']:.3f} in [0.20, 0.35)) "
                 f"corroborated by Branch B (Z_diff={it['z_diff']:.2f} >= {Z_REVIEW})", it)
        return _pack(3, "REVIEW", "RECALL_RESCUE", reasons, details)

    # --- Level 3b: REVIEW open-set (missed by YOLO, large Branch B anomaly) -
    review3b = _has(lambda it: (
        it.get("source") == "branchb"
        and it.get("z_diff") is not None
        and it["z_diff"] >= Z_OPENSET
        and it.get("area") is not None
        and it["area"] >= AREA_OPENSET))
    if review3b:
        for it in review3b:
            note(3, "REVIEW", "UNKNOWN_DEFECT",
                 f"Branch B structural anomaly (Z_diff={it['z_diff']:.2f} >= {Z_OPENSET}, "
                 f"Area={it['area']:.0f} px >= {AREA_OPENSET}) not classified by YOLO", it)
        return _pack(3, "REVIEW", "UNKNOWN_DEFECT", reasons, details)

    # --- Level 4: PASS / NONE ----------------------------------------------
    for it in items:
        note(4, "PASS", "NONE",
              "Candidate region below FAIL / REVIEW thresholds", it)
    return _pack(4, "PASS", "NONE", reasons, details)


def _pack(level: int, conclusion: str, verdict: str,
          reasons: list[str], details: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "level": level,
        "conclusion": conclusion,
        "verdict": verdict,
        "reasons": reasons,
        "details": details,
    }


def summary_text(res: dict[str, Any]) -> str:
    return f"{res['conclusion']} / {res['verdict']} (level {res['level']})"
