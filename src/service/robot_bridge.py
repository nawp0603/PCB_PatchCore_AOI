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

BINS = ("FAIL", "PASS")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class RobotBridge:
    def __init__(self, root: Path, ttl_s: float = 30.0):
        self.outbox = Path(root) / "outbox"
        self.inbox = Path(root) / "inbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s

    def send(self, bin_name: str, conclusion: str, decided_by: str, source: str,
             now: float | None = None) -> str:
        if bin_name not in BINS:
            raise ValueError(f"Unknown bin: {bin_name}")
        now = time.time() if now is None else now
        inspection_id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{uuid.uuid4().hex[:6]}"
        _write_json(self.outbox / f"{inspection_id}.json", {
            "inspection_id": inspection_id, "bin": bin_name, "conclusion": conclusion,
            "decided_by": decided_by, "source": source,
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


def process_pending(bridge: RobotBridge, driver, now=time.time) -> list[tuple[str, str]]:
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
            driver.sort(command["bin"])
        except Exception as exc:
            bridge.ack(inspection_id, "error", f"{type(exc).__name__}: {exc}")
            handled.append((inspection_id, "error"))
            raise
        bridge.ack(inspection_id, "done")
        handled.append((inspection_id, "done"))
    return handled
