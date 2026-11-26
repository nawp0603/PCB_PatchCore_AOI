import json

import pytest

from src.service.robot_bridge import RobotBridge, process_pending


class FakeDriver:
    def __init__(self, fail=False):
        self.sorted, self.fail = [], fail

    def sort(self, source_slot, bin_name):
        if self.fail:
            raise RuntimeError("arm stuck")
        self.sorted.append((source_slot, bin_name))


def test_command_round_trip(tmp_path):
    bridge = RobotBridge(tmp_path)
    job = bridge.send("C11", "FAIL", "auto", "A_05.png", now=100.0)
    command = json.loads((tmp_path / "outbox" / f"{job}.json").read_text(encoding="utf-8"))
    assert command["bin"] == "C11" and command["expires"] == 130.0
    assert bridge.poll(job) is None
    driver = FakeDriver()
    assert process_pending(bridge, driver, now=lambda: 110.0) == [(job, "done")]
    assert driver.sorted == [("B11", "C11")]
    assert bridge.poll(job)["status"] == "done"
    assert process_pending(bridge, driver, now=lambda: 111.0) == []   # executed only once
    assert not list(tmp_path.rglob("*.tmp"))


def test_expired_command_is_never_executed(tmp_path):
    bridge = RobotBridge(tmp_path)
    job = bridge.send("A11", "KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "auto", "x", now=100.0)
    driver = FakeDriver()
    assert process_pending(bridge, driver, now=lambda: 131.0) == [(job, "expired")]
    assert driver.sorted == []


def test_driver_error_is_acknowledged_and_stops(tmp_path):
    bridge = RobotBridge(tmp_path)
    first = bridge.send("C11", "FAIL", "auto", "a", now=100.0)
    second = bridge.send("A11", "REVIEW", "operator", "b", now=101.0)
    with pytest.raises(RuntimeError):
        process_pending(bridge, FakeDriver(fail=True), now=lambda: 102.0)
    assert bridge.poll(first)["status"] == "error"
    assert bridge.poll(second) is None


def test_unknown_bin_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        RobotBridge(tmp_path).send("REVIEW", "REVIEW", "auto", "x")


@pytest.mark.parametrize("legacy_slot", ["B1", "B2", "B12"])
def test_legacy_source_slots_are_canonicalized_to_b11(tmp_path, legacy_slot):
    bridge = RobotBridge(tmp_path)
    job = bridge.send("C11", "FAIL", "auto", "x", source_slot=legacy_slot, now=100.0)

    assert bridge.pending()[0]["source_slot"] == "B11"
    driver = FakeDriver()
    process_pending(bridge, driver, now=lambda: 101.0)
    assert driver.sorted == [("B11", "C11")]


def test_status_publishes_worker_detail(tmp_path):
    bridge = RobotBridge(tmp_path)
    bridge.report_status("busy", "moving to C11")

    assert bridge.status()["detail"] == "moving to C11"


def test_control_round_trip(tmp_path):
    bridge = RobotBridge(tmp_path)
    control = bridge.send_control("read_pose", now=100.0)
    assert bridge.pending_controls()[0]["action"] == "read_pose"
    bridge.ack_control(control, "done", {"pose": [1, 2, 3, 4]})
    assert bridge.poll_control(control)["result"]["pose"] == [1, 2, 3, 4]
