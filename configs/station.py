"""Canonical station names and in-memory migration for deployed configurations."""

SOURCE_SLOTS = ("B11",)
TEACH_POINTS = ("Home", "B11", "A11", "C11", "C12")
PASS_SLOTS = ("A11", "A12", "A13")
FAIL_SLOTS = ("C11", "C12", "C13")
LEGACY_SOURCE_SLOTS = {"B1": "B11", "B2": "B11", "B12": "B11"}


def canonical_source_slot(slot: str) -> str:
    """Accept legacy command files while always targeting the current physical slot."""
    return LEGACY_SOURCE_SLOTS.get(slot, slot)


def migrate_station_config(config: dict) -> bool:
    """Upgrade old source names and discard retired camera-slot calibration."""
    changed = False
    for section in ("pick_positions",):
        values = config.get(section)
        if not isinstance(values, dict):
            continue
        for legacy, current in LEGACY_SOURCE_SLOTS.items():
            if legacy in values:
                if current not in values:
                    values[current] = values[legacy]
                del values[legacy]
                changed = True
        if "B12" in values:
            del values["B12"]
            changed = True
    bins = config.get("bins")
    if isinstance(bins, dict) and "A" in bins:
        if "A11" not in bins:
            bins["A11"] = bins["A"]
        del bins["A"]
        changed = True
    for section in ("camera_slots", "slot_occupancy", "empty_station_reference"):
        if section in config:
            del config[section]
            changed = True
    return changed


def derive_drop_slots(config: dict) -> bool:
    """Generate stacked A/C drop positions from the taught C11 -> C12 vector."""
    picks, bins = config.get("pick_positions"), config.get("bins")
    if not isinstance(picks, dict) or not isinstance(bins, dict):
        return False
    try:
        b11 = [float(value) for value in picks["B11"]]
        a11 = [float(value) for value in bins["A11"]]
        c11 = [float(value) for value in bins["C11"]]
        c12 = [float(value) for value in bins["C12"]]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(len(pose) == 4 for pose in (b11, a11, c11, c12)):
        return False
    delta = [c12[index] - c11[index] for index in range(3)]
    pickup_r = b11[3]
    changed = False
    for base_name, base_pose, names in (("A11", a11, PASS_SLOTS), ("C11", c11, FAIL_SLOTS)):
        for index, name in enumerate(names):
            pose = [base_pose[axis] + delta[axis] * index for axis in range(3)] + [pickup_r]
            if bins.get(name) != pose:
                bins[name] = pose
                changed = True
    return changed
