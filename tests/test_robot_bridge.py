import json

import pytest

from src.service.robot_bridge import RobotBridge, process_pending


class FakeDriver:
    def __init__(self, fail=False):
        self.sorted, self.fail = [], fail

    def sort(self, bin_name):
        if self.fail:
            raise RuntimeError("arm stuck")
        self.sorted.append(bin_name)


def test_command_round_trip(tmp_path):
    bridge = RobotBridge(tmp_path)
    job = bridge.send("FAIL", "FAIL", "auto", "A_05.png", now=100.0)
    command = json.loads((tmp_path / "outbox" / f"{job}.json").read_text(encoding="utf-8"))
    assert command["bin"] == "FAIL" and command["expires"] == 130.0
    assert bridge.poll(job) is None
    driver = FakeDriver()
    assert process_pending(bridge, driver, now=lambda: 110.0) == [(job, "done")]
    assert driver.sorted == ["FAIL"]
    assert bridge.poll(job)["status"] == "done"
    assert process_pending(bridge, driver, now=lambda: 111.0) == []   # executed only once
    assert not list(tmp_path.rglob("*.tmp"))


def test_expired_command_is_never_executed(tmp_path):
    bridge = RobotBridge(tmp_path)
    job = bridge.send("PASS", "KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "auto", "x", now=100.0)
    driver = FakeDriver()
    assert process_pending(bridge, driver, now=lambda: 131.0) == [(job, "expired")]
    assert driver.sorted == []


def test_driver_error_is_acknowledged_and_stops(tmp_path):
    bridge = RobotBridge(tmp_path)
    first = bridge.send("FAIL", "FAIL", "auto", "a", now=100.0)
    second = bridge.send("PASS", "REVIEW", "operator", "b", now=101.0)
    with pytest.raises(RuntimeError):
        process_pending(bridge, FakeDriver(fail=True), now=lambda: 102.0)
    assert bridge.poll(first)["status"] == "error"
    assert bridge.poll(second) is None


def test_unknown_bin_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        RobotBridge(tmp_path).send("REVIEW", "REVIEW", "auto", "x")
