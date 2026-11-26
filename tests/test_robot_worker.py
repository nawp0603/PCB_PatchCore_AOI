from configs.station import derive_drop_slots, migrate_station_config
from unittest.mock import Mock

from scripts.robot_worker import DobotDriver, validate_config


def test_direct_serial_packet_uses_standard_twos_complement_checksum():
    packet = DobotDriver._packet(DobotDriver.SET_PTP_CMD, 0x03, b"\x02\x01\x02")

    assert packet[:5] == b"\xaa\xaa\x05\x54\x03"
    assert packet[-1] == ((-sum(packet[3:-1])) & 0xFF)


def test_station_config_requires_both_logical_destinations():
    config = {"port": "COM15", "home": [0, 0, 1, 0], "pick_positions": {"B11": [1, 1, 0, 0]},
              "bins": {"C11": [2, 2, 0, 0]}, "safe_z": 1}

    try:
        validate_config(config)
    except ValueError as exc:
        assert "A11" in str(exc)
    else:
        raise AssertionError("missing PASS position must be rejected")


def test_legacy_station_positions_migrate_to_physical_slots():
    config = {"pick_positions": {"B12": [1, 2, 3, 4]}, "bins": {"A": [5, 6, 7, 8]},
              "camera_slots": {"B12": [0, 0, 0, 0]}}
    assert migrate_station_config(config)
    assert config["pick_positions"] == {"B11": [1, 2, 3, 4]}
    assert config["bins"]["A11"] == [5, 6, 7, 8]
    assert "camera_slots" not in config


def test_drop_slots_use_the_c11_c12_stack_vector_and_pick_rotation():
    config = {"pick_positions": {"B11": [0, 0, 0, 7]},
              "bins": {"A11": [1, 2, 3, 0], "C11": [4, 5, 6, 0], "C12": [5, 7, 9, 0]}}
    assert derive_drop_slots(config)
    assert config["bins"]["A12"] == [2.0, 4.0, 6.0, 7.0]
    assert config["bins"]["C13"] == [6.0, 9.0, 12.0, 7.0]


def test_close_stops_queue_clears_queue_and_releases_suction_before_serial_close():
    driver = DobotDriver.__new__(DobotDriver)
    driver._write = Mock()
    driver.ser = Mock()

    driver.close()

    assert [call.args[:2] for call in driver._write.call_args_list] == [
        (DobotDriver.FORCE_STOP_QUEUE, 0x01),
        (DobotDriver.CLEAR_QUEUE, 0x01),
        (DobotDriver.SET_SUCTION, 0x01),
    ]
    driver.ser.close.assert_called_once()


def test_release_is_immediate_and_repeated_before_the_next_lift(monkeypatch):
    driver = DobotDriver.__new__(DobotDriver)
    driver.cfg = {"suction_settle_s": 0.0, "release_settle_s": 0.0}
    driver._write = Mock()
    monkeypatch.setattr("scripts.robot_worker.time.sleep", lambda _seconds: None)

    driver._suck(False)

    assert [call.args[:2] for call in driver._write.call_args_list] == [
        (DobotDriver.SET_SUCTION, 0x01),
        (DobotDriver.SET_SUCTION, 0x01),
    ]
