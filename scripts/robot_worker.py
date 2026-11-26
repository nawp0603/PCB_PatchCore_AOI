"""Robot side of the sorting demo: run next to the inspection UI.

    python scripts/robot_worker.py --sim                 # no robot, prints and waits 2 s per board
    python scripts/robot_worker.py --config configs/robot_positions.json

The worker picks the board at the fixed jig and drops it into the FAIL or PASS bin for every
command the UI writes to robot_io/outbox, then acknowledges in robot_io/inbox. The arm must be
back at "home" (out of the camera view) before the acknowledgement, because the UI only allows
the next capture after it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.robot_bridge import RobotBridge, process_pending  # noqa: E402
from configs.station import FAIL_SLOTS, PASS_SLOTS, SOURCE_SLOTS, derive_drop_slots, migrate_station_config  # noqa: E402


def _pose(config: dict, name: str) -> tuple[float, float, float, float]:
    value = config.get(name)
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be [x, y, z, r].")
    try:
        pose = tuple(float(number) for number in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not all(math.isfinite(number) for number in pose):
        raise ValueError(f"{name} must contain finite values.")
    return pose


def validate_config(config: dict) -> None:
    if not isinstance(config.get("port"), str) or not config["port"].strip():
        raise ValueError("port must be a non-empty serial port name, for example COM15.")
    _pose(config, "home")
    picks = config.get("pick_positions")
    if not isinstance(picks, dict):
        raise ValueError("pick_positions must define the B11 pose.")
    poses = [_pose(picks, slot) for slot in SOURCE_SLOTS]
    bins = config.get("bins")
    if not isinstance(bins, dict):
        raise ValueError("bins must define A11, C11, C12 and C13 poses.")
    for bin_name in (*PASS_SLOTS, *FAIL_SLOTS):
        poses.append(_pose(bins, bin_name))
    try:
        safe_z = float(config["safe_z"])
        settle_s = float(config.get("suction_settle_s", 0.5))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("safe_z and suction_settle_s must be numeric.") from exc
    if not math.isfinite(safe_z) or safe_z < max(pose[2] for pose in poses):
        raise ValueError("safe_z must be a finite height at or above every taught pose.")
    if not math.isfinite(settle_s) or settle_s < 0:
        raise ValueError("suction_settle_s must be zero or greater.")
    for name in ("ptp_speed_ratio", "ptp_acceleration_ratio"):
        ratio = float(config.get(name, 50.0))
        if not math.isfinite(ratio) or not 1 <= ratio <= 100:
            raise ValueError(f"{name} must be between 1 and 100.")


def acquire_worker_lock(io_root: Path):
    """Allow only one worker to control the station serial port."""
    if os.name != "nt":
        return None
    import msvcrt
    lock = (io_root / "worker.lock").open("a+b")
    lock.seek(0)
    if lock.read(1) == b"":
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        return None
    return lock


def release_worker_lock(lock) -> None:
    if lock is None:
        return
    import msvcrt
    try:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        lock.close()


class SimDriver:
    def __init__(self, config: dict, seconds: float = 2.0):
        self.config = config
        self.seconds = seconds
        self.current_pose = list(config["home"])

    def home(self) -> None:
        self.current_pose = list(self.config["home"])
        print("[SIM] home")

    def move(self, pose) -> None:
        self.current_pose = list(pose)
        print(f"[SIM] move -> {self.current_pose}")

    def get_pose(self):
        return list(self.current_pose)

    def sort(self, source_slot: str, bin_name: str) -> None:
        print(f"[SIM] gap bo tai {source_slot} -> khay {bin_name}")
        time.sleep(self.seconds)
        print("[SIM] ve vi tri cho")

    def close(self) -> None:
        pass


class DobotDriver:
    """Direct serial driver for the Dobot Magician Lite protocol.

    This talks to the robot's COM port directly; DobotLab and DobotLink are not used.
    """

    GET_POSE = 10
    SET_SUCTION = 62
    SET_PTP_JOINT_PARAMS = 80
    SET_PTP_COORDINATE_PARAMS = 81
    SET_PTP_JUMP_PARAMS = 82
    SET_PTP_COMMON_PARAMS = 83
    SET_PTP_CMD = 84
    GET_ALARMS = 20
    CLEAR_ALARMS = 20
    START_QUEUE = 240
    CLEAR_QUEUE = 245
    FORCE_STOP_QUEUE = 242
    GET_QUEUE_INDEX = 246
    PTP_MOVJ_XYZ = 1
    PTP_MOVL_XYZ = 2

    def __init__(self, config: dict):
        import serial

        self.cfg = config
        self.ser = serial.Serial(
            port=config["port"], baudrate=115200, timeout=0.05, write_timeout=1.0)
        self._lock = threading.Lock()
        self._rx = bytearray()
        self.ser.reset_input_buffer()
        # Lite accepts setup commands only as non-queued writes. Do not use the
        # Magician legacy PTP parameter commands (80--82) on this controller.
        self._write(self.FORCE_STOP_QUEUE, 0x01)
        self._write(self.CLEAR_QUEUE, 0x01)
        alarms = self._command(self.GET_ALARMS, 0x00)
        if any(alarms):
            self._write(self.CLEAR_ALARMS, 0x01)
            time.sleep(0.1)
            alarms = self._command(self.GET_ALARMS, 0x00)
            if any(alarms):
                raise RuntimeError(f"Dobot vẫn còn alarm: {alarms.hex(' ')}")
        self._write(self.SET_PTP_COMMON_PARAMS, 0x01, struct.pack(
            "<2f", float(config.get("ptp_speed_ratio", 50.0)),
            float(config.get("ptp_acceleration_ratio", 50.0))))
        self._write(self.START_QUEUE, 0x01)
        self.get_pose()

    @staticmethod
    def _packet(command_id: int, control: int, params: bytes = b"") -> bytes:
        body = bytes((command_id, control)) + params
        # The Lite uses the standard two's-complement checksum, including checksum 255.
        return b"\xaa\xaa" + bytes((len(body),)) + body + bytes(((-sum(body)) & 0xFF,))

    def _read_packet(self, expected_id: int, timeout: float = 1.5) -> bytes:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            waiting = self.ser.in_waiting
            if waiting:
                self._rx.extend(self.ser.read(waiting))
            while True:
                header = self._rx.find(b"\xaa\xaa")
                if header >= 0:
                    if header:
                        del self._rx[:header]
                    if len(self._rx) < 3:
                        break
                    packet_size = self._rx[2] + 4
                    if len(self._rx) < packet_size:
                        break
                    packet = bytes(self._rx[:packet_size])
                    del self._rx[:packet_size]
                    # Lite firmware variants use different checksum edge-case rules.
                    # Header, declared length, and command ID are enough to frame replies.
                    if packet[3] == expected_id:
                        return packet[5:-1]
                else:
                    self._rx.clear()
                    break
            time.sleep(0.01)
        raise TimeoutError(f"Dobot không trả lời lệnh serial {expected_id}.")

    def _write(self, command_id: int, control: int, params: bytes = b"") -> None:
        with self._lock:
            self.ser.write(self._packet(command_id, control, params))
            self.ser.flush()

    def _command(self, command_id: int, control: int, params: bytes = b"", timeout: float = 1.5) -> bytes:
        with self._lock:
            self.ser.write(self._packet(command_id, control, params))
            self.ser.flush()
            return self._read_packet(command_id, timeout)

    def _queue_index(self) -> int:
        response = self._command(self.GET_QUEUE_INDEX, 0x00)
        if len(response) < 8:
            raise RuntimeError("Dobot trả chỉ số hàng đợi không hợp lệ.")
        return struct.unpack_from("<Q", response)[0]

    def _move(self, x, y, z, r=0.0, linear=False) -> None:
        # pydobot's queued-command index may be returned before the physical arm
        # reaches the point. Track the reported pose instead of failing mid-motion.
        mode = self.PTP_MOVL_XYZ if linear else self.PTP_MOVJ_XYZ
        response = self._command(self.SET_PTP_CMD, 0x03, struct.pack("<B4f", mode, x, y, z, r))
        if len(response) < 8:
            raise RuntimeError("Dobot không xác nhận lệnh di chuyển.")
        expected_index = struct.unpack_from("<Q", response)[0]
        queue_deadline = time.monotonic() + 30.0
        while time.monotonic() < queue_deadline:
            if self._queue_index() == expected_index:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Dobot không hoàn tất lệnh hàng đợi {expected_index} trong 30 s.")
        target = (float(x), float(y), float(z), float(r))
        deadline = time.monotonic() + 30.0
        actual = None
        while time.monotonic() < deadline:
            actual = self.get_pose()
            if all(abs(measured - expected) <= 3.0 for measured, expected in zip(actual, target)):
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"Dobot did not reach target {target} within 30 s; "
            f"reported pose {tuple(round(v, 2) for v in (actual or ())) }.")

    def _suck(self, on: bool) -> None:
        params = bytes((1, int(on)))
        if on:
            # Queue pickup after the confirmed descent at B11.
            self._write(self.SET_SUCTION, 0x03, params)
            time.sleep(self.cfg.get("suction_settle_s", 0.5))
            return
        # Release must not remain behind a queued motion: issue it immediately,
        # then repeat once so the valve has a deterministic off command before lift.
        self._write(self.SET_SUCTION, 0x01, params)
        time.sleep(0.05)
        self._write(self.SET_SUCTION, 0x01, params)
        time.sleep(self.cfg.get("release_settle_s", self.cfg.get("suction_settle_s", 0.5)))

    def home(self) -> None:
        self._move(*self.cfg["home"])

    def move(self, pose) -> None:
        self._move(*pose)

    def get_pose(self):
        response = self._command(self.GET_POSE, 0x00)
        if len(response) < 16:
            raise RuntimeError("Dobot trả tọa độ không hợp lệ.")
        return list(struct.unpack_from("<4f", response))

    def sort(self, source_slot: str, bin_name: str) -> None:
        safe_z = self.cfg["safe_z"]
        px, py, pz, pr = self.cfg["pick_positions"][source_slot]
        bx, by, bz, br = self.cfg["bins"][bin_name]
        self._move(px, py, safe_z, pr)
        self._move(px, py, pz, pr, linear=True)
        self._suck(True)
        self._move(px, py, safe_z, pr)
        self._move(bx, by, safe_z, br)
        self._move(bx, by, bz, br, linear=True)
        self._suck(False)
        self._move(bx, by, safe_z, br)
        self.home()

    def close(self) -> None:
        # Never leave queued motion or suction active when this process exits.
        for command_id, control, params in (
                (self.FORCE_STOP_QUEUE, 0x01, b""),
                (self.CLEAR_QUEUE, 0x01, b""),
                (self.SET_SUCTION, 0x01, bytes((1, 0)))):
            try:
                self._write(command_id, control, params)
            except Exception:
                pass
        self.ser.close()


def process_controls(bridge: RobotBridge, driver, now=time.time, on_before=None) -> list[tuple[str, str]]:
    """Process UI teaching controls while the robot is idle."""
    handled = []
    for control in bridge.pending_controls():
        control_id = control["control_id"]
        if now() > control["expires"]:
            bridge.ack_control(control_id, "expired")
            handled.append((control_id, "expired"))
            continue
        try:
            if on_before is not None:
                on_before(control)
            if control["action"] == "read_pose":
                bridge.ack_control(control_id, "done", {"pose": driver.get_pose()})
            elif control["action"] == "home":
                driver.home()
                bridge.ack_control(control_id, "done", {"pose": driver.get_pose()})
            elif control["action"] == "move":
                pose = control.get("payload", {}).get("pose")
                if not isinstance(pose, list) or len(pose) != 4:
                    raise ValueError("move control needs pose [x, y, z, r].")
                driver.move([float(value) for value in pose])
                bridge.ack_control(control_id, "done", {"pose": driver.get_pose()})
            else:  # RobotBridge rejects unknown actions, retained for malformed files.
                raise ValueError(f"Unknown control action: {control['action']}")
            handled.append((control_id, "done"))
        except Exception as exc:
            bridge.ack_control(control_id, "error", error=f"{type(exc).__name__}: {exc}")
            handled.append((control_id, "error"))
            # A transient pose-read failure must not terminate the production worker.
            if control["action"] != "read_pose":
                raise
    return handled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--io", type=Path, default=ROOT / "robot_io")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "robot_positions.json")
    parser.add_argument("--port", help="Override the serial port selected from the AOI UI.")
    parser.add_argument("--sim", action="store_true", help="No robot: print and wait.")
    parser.add_argument("--teach", action="store_true",
                        help="Connect for coordinate teaching only; do not home or sort PCBs.")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate the robot configuration without connecting or moving.")
    args = parser.parse_args()

    bridge = RobotBridge(args.io)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    migrated = migrate_station_config(config)
    migrated = derive_drop_slots(config) or migrated
    if migrated:
        args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.port:
        config["port"] = args.port
    try:
        validate_config(config)
    except ValueError as exc:
        bridge.report_status("error", str(exc))
        print(f"CAU HINH ROBOT KHONG HOP LE: {exc}")
        return 1
    if args.check_config:
        calibrated = bool(config.get("station_calibrated"))
        print(f"Cau hinh hop le: port={config['port']}; station_calibrated={calibrated}")
        return 0
    if not args.sim and not args.teach and not config.get("station_calibrated", False):
        raise RuntimeError(
            "Robot that is locked: teach every pose, then set station_calibrated to true.")

    worker_lock = acquire_worker_lock(bridge.root)
    if os.name == "nt" and worker_lock is None:
        print("ROBOT KHONG KHOI DONG: Mot worker khac dang dung cong COM nay.")
        return 1
    bridge.report_status("connecting")
    # Commands left over from an earlier session are stale: expire them, never execute.
    for command in bridge.pending():
        bridge.ack(command["inspection_id"], "expired")
    for control in bridge.pending_controls():
        bridge.ack_control(control["control_id"], "expired")
    driver = None
    failed = False
    try:
        driver = SimDriver(config) if args.sim else DobotDriver(config)
        if not args.teach:
            driver.home()
        bridge.report_status("ready")
        last_heartbeat = time.monotonic()
        print("Robot san sang de day toa do." if args.teach else
              f"Robot san sang. Doc lenh tu {bridge.outbox} (Ctrl+C de dung).")
        while True:
            def mark_busy(_command):
                bridge.report_status("busy")

            control_handled = process_controls(bridge, driver, on_before=mark_busy)
            if args.teach:
                handled = []
                for command in bridge.pending():
                    bridge.ack(command["inspection_id"], "expired")
            else:
                handled = process_pending(bridge, driver, on_before=mark_busy)
            for inspection_id, status in handled:
                print(f"{inspection_id}: {status}")
            if control_handled or handled:
                bridge.report_status("ready")
                last_heartbeat = time.monotonic()
            elif time.monotonic() - last_heartbeat >= 1.0:
                bridge.report_status("ready")
                last_heartbeat = time.monotonic()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Da dung.")
        return 0
    except Exception as exc:
        failed = True
        bridge.report_status("error", f"{type(exc).__name__}: {exc}")
        print(f"ROBOT DUNG DO LOI: {type(exc).__name__}: {exc}")
        print("Kiem tra canh tay va bo mach, roi chay lai.")
        return 1
    finally:
        if driver is not None:
            driver.close()
        if not failed:
            bridge.report_status("offline")
        release_worker_lock(worker_lock)


if __name__ == "__main__":
    raise SystemExit(main())
