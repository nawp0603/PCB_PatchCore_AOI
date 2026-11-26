"""File handshake between the inspection UI and the robot worker.

UI writes   <root>/outbox/<id>.json : {"inspection_id", "bin", "conclusion",
                                       "decided_by", "source", "created", "expires"}
Robot writes <root>/inbox/<id>.json : {"inspection_id", "status": "done" | "error" | "expired",
                                       "error"?, "finished"}
Each file is written to a .tmp name first and then renamed, so a reader never sees half a file.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from configs.station import FAIL_SLOTS, PASS_SLOTS, SOURCE_SLOTS, canonical_source_slot

BINS = (*PASS_SLOTS, *FAIL_SLOTS)
CONTROL_ACTIONS = ("read_pose", "move", "home")


def _write_json(path: Path, data: dict) -> None:
    # Windows can briefly lock status.json while the UI polls it.  A unique temporary
    # name also prevents two writers from overwriting each other's staging file.
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.02 * (attempt + 1))
    raise RuntimeError("Không thể ghi trạng thái robot.")


class RobotBridge:
    def __init__(self, root: Path, ttl_s: float = 30.0):
        self.root = Path(root)
        self.outbox = self.root / "outbox"
        self.inbox = self.root / "inbox"
        self.control_outbox = self.root / "control_outbox"
        self.control_inbox = self.root / "control_inbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.control_outbox.mkdir(parents=True, exist_ok=True)
        self.control_inbox.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s

    def report_status(self, status: str, detail: str = "") -> None:
        data = {"status": status, "updated": time.time()}
        if detail:
            data["detail"] = detail
        _write_json(self.root / "status.json", data)

    def status(self) -> dict | None:
        path = self.root / "status.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def send_control(self, action: str, payload: dict | None = None,
                     now: float | None = None) -> str:
        if action not in CONTROL_ACTIONS:
            raise ValueError(f"Unknown robot control action: {action}")
        now = time.time() if now is None else now
        control_id = f"control-{uuid.uuid4().hex[:10]}"
        _write_json(self.control_outbox / f"{control_id}.json", {
            "control_id": control_id, "action": action, "payload": payload or {},
            "created": now, "expires": now + self.ttl_s})
        return control_id

    def poll_control(self, control_id: str) -> dict | None:
        path = self.control_inbox / f"{control_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def pending_controls(self) -> list[dict]:
        done = {path.stem for path in self.control_inbox.glob("*.json")}
        controls = [json.loads(path.read_text(encoding="utf-8"))
                    for path in self.control_outbox.glob("*.json") if path.stem not in done]
        return sorted(controls, key=lambda control: control["created"])

    def ack_control(self, control_id: str, status: str, result: dict | None = None,
                    error: str | None = None) -> None:
        data = {"control_id": control_id, "status": status, "finished": time.time()}
        if result is not None:
            data["result"] = result
        if error:
            data["error"] = error
        _write_json(self.control_inbox / f"{control_id}.json", data)

    def send(self, bin_name: str, conclusion: str, decided_by: str, source: str,
             source_slot: str = "B11", now: float | None = None) -> str:
        if bin_name not in BINS:
            raise ValueError(f"Unknown bin: {bin_name}")
        source_slot = canonical_source_slot(source_slot)
        if source_slot not in SOURCE_SLOTS:
            raise ValueError(f"Unknown source slot: {source_slot}")
        now = time.time() if now is None else now
        inspection_id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{uuid.uuid4().hex[:6]}"
        _write_json(self.outbox / f"{inspection_id}.json", {
            "inspection_id": inspection_id, "bin": bin_name, "conclusion": conclusion,
            "decided_by": decided_by, "source": source, "source_slot": source_slot,
            "created": now, "expires": now + self.ttl_s})
        return inspection_id

    def poll(self, inspection_id: str) -> dict | None:
        path = self.inbox / f"{inspection_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def pending(self) -> list[dict]:
        done = {path.stem for path in self.inbox.glob("*.json")}
        commands = [json.loads(path.read_text(encoding="utf-8"))
                    for path in self.outbox.glob("*.json") if path.stem not in done]
        return sorted(commands, key=lambda command: command["created"])

    def ack(self, inspection_id: str, status: str, error: str | None = None) -> None:
        data = {"inspection_id": inspection_id, "status": status, "finished": time.time()}
        if error:
            data["error"] = error
        _write_json(self.inbox / f"{inspection_id}.json", data)


def process_pending(bridge: RobotBridge, driver, now=time.time, on_before=None) -> list[tuple[str, str]]:
    """Execute every unacknowledged command once, oldest first.

    A command past its expiry is never executed (the UI has given up on it and the board
    may have changed). A driver error stops processing so a person can check the arm.
    """
    handled = []
    for command in bridge.pending():
        inspection_id = command["inspection_id"]
        if now() > command["expires"]:
            bridge.ack(inspection_id, "expired")
            handled.append((inspection_id, "expired"))
            continue
        try:
            if on_before is not None:
                on_before(command)
            driver.sort(canonical_source_slot(command.get("source_slot", "B11")), command["bin"])
        except Exception as exc:
            bridge.ack(inspection_id, "error", f"{type(exc).__name__}: {exc}")
            handled.append((inspection_id, "error"))
            raise
        bridge.ack(inspection_id, "done")
        handled.append((inspection_id, "done"))
    return handled
