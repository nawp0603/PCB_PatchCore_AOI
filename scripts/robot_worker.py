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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.robot_bridge import RobotBridge, process_pending  # noqa: E402


class SimDriver:
    def __init__(self, seconds: float = 2.0):
        self.seconds = seconds

    def home(self) -> None:
        print("[SIM] home")

    def sort(self, bin_name: str) -> None:
        print(f"[SIM] gap bo tai cu -> khay {bin_name}")
        time.sleep(self.seconds)
        print("[SIM] ve vi tri cho")

    def close(self) -> None:
        pass


class DobotDriver:
    """Suction-cup pick and place with pydobot (pip install pydobot).

    pydobot implements the Dobot Magician serial protocol. If it cannot talk to the
    Magician Lite, replace the body of _move/_suck with the official Dobot API; nothing
    else in the demo depends on this class.
    """

    def __init__(self, config: dict):
        from pydobot import Dobot

        self.cfg = config
        self.device = Dobot(port=config["port"], verbose=False)

    def _move(self, x, y, z, r=0.0) -> None:
        self.device.move_to(x, y, z, r, wait=True)

    def _suck(self, on: bool) -> None:
        self.device.suck(on)
        time.sleep(self.cfg.get("suction_settle_s", 0.5))

    def home(self) -> None:
        self._move(*self.cfg["home"])

    def sort(self, bin_name: str) -> None:
        safe_z = self.cfg["safe_z"]
        px, py, pz, pr = self.cfg["pick"]
        bx, by, bz, br = self.cfg["bins"][bin_name]
        self._move(px, py, safe_z, pr)
        self._move(px, py, pz, pr)
        self._suck(True)
        self._move(px, py, safe_z, pr)
        self._move(bx, by, safe_z, br)
        self._move(bx, by, bz, br)
        self._suck(False)
        self._move(bx, by, safe_z, br)
        self.home()

    def close(self) -> None:
        self.device.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--io", type=Path, default=ROOT / "robot_io")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "robot_positions.json")
    parser.add_argument("--sim", action="store_true", help="No robot: print and wait.")
    args = parser.parse_args()

    bridge = RobotBridge(args.io)
    # Commands left over from an earlier session are stale: expire them, never execute.
    for command in bridge.pending():
        bridge.ack(command["inspection_id"], "expired")
    driver = SimDriver() if args.sim else DobotDriver(
        json.loads(args.config.read_text(encoding="utf-8")))
    driver.home()
    print(f"Robot san sang. Doc lenh tu {bridge.outbox} (Ctrl+C de dung).")
    try:
        while True:
            for inspection_id, status in process_pending(bridge, driver):
                print(f"{inspection_id}: {status}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Da dung.")
        return 0
    except Exception as exc:
        print(f"ROBOT DUNG DO LOI: {type(exc).__name__}: {exc}")
        print("Kiem tra canh tay va bo mach, roi chay lai.")
        return 1
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
