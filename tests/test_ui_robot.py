from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import numpy as np
import pytest

from src.ui.tkinter_app import TkAOIApp, _pcb_present, _use_branch_b


@pytest.fixture
def app(monkeypatch):
    app = TkAOIApp.__new__(TkAOIApp)
    app.robot = Mock()
    app.robot.send.return_value = "job-1"
    app._robot_job = None
    app._robot_delay_job = None
    app._robot_delay_after_id = None
    app._automation_after_id = None
    app._automation_phase = "idle"
    app.root = Mock()
    app.lbl_robot = Mock()
    app._set_status = Mock()
    monkeypatch.setattr("src.ui.tkinter_app.messagebox.showwarning", Mock())
    monkeypatch.setattr("src.ui.tkinter_app.messagebox.showerror", Mock())
    return app


@pytest.mark.parametrize(("conclusion", "bin_name"), [
    ("FAIL", "C11"), ("KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "A11")])
def test_auto_pick_waits_five_seconds_before_sending(app, conclusion, bin_name):
    items = [{"source": "yolo", "class_id": 0}] if conclusion == "FAIL" else []
    app._after_result({"conclusion": conclusion, "items": items}, "GigE camera:B12")

    assert app._robot_delay_job["bin"] == bin_name
    assert app._robot_blocking()
    app.robot.send.assert_not_called()

    app._dispatch_delayed_robot_pick()

    app.robot.send.assert_called_once_with(bin_name, conclusion, "auto", "GigE camera:B12", "B11")
    assert app._robot_job["id"] == "job-1"


@pytest.mark.parametrize("conclusion", ["REVIEW", "CONFIRM", "RECAPTURE"])
def test_non_pick_conclusions_never_offer_or_send_pick(app, conclusion):
    app._after_result({"conclusion": conclusion}, "a.png")

    app.robot.send.assert_not_called()
    assert not app._robot_blocking()


def test_empty_frame_never_schedules_a_robot_pick(app):
    app._after_result({"conclusion": "KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "pcb_present": False}, "GigE camera:B11")

    app.robot.send.assert_not_called()
    assert not app._robot_blocking()


def test_automation_stops_when_no_pcb_is_detected(app):
    app._automation_enabled = True
    app._stop_automation = Mock()

    app._automation_after_result({"pcb_present": False}, "GigE camera:B11")

    app._stop_automation.assert_called_once()


def test_pcb_presence_requires_a_substantial_green_board_region():
    empty = np.zeros((400, 400, 3), np.uint8)
    board = empty.copy()
    board[80:320, 80:320] = (20, 100, 20)

    assert not _pcb_present(empty)
    assert _pcb_present(board)


def test_rework_without_routable_yolo_class_does_not_pick(app):
    app._after_result({"conclusion": "REWORK", "items": [{"source": "patchcore"}]}, "a.png")

    app.robot.send.assert_not_called()


def test_ack_unblocks(app):
    app._after_result({"conclusion": "FAIL", "items": [{"source": "yolo", "class_id": 0}]}, "a.png")
    app._dispatch_delayed_robot_pick()
    app.robot.poll.return_value = {"status": "done"}

    app._robot_poll()

    assert app._robot_job is None and not app._robot_blocking()


def test_automation_schedules_the_next_capture_after_robot_ack(app):
    app._automation_enabled = True
    app._automation_schedule = Mock()
    app._after_result({"conclusion": "FAIL", "items": [{"source": "yolo", "class_id": 0}]}, "a.png")
    app._dispatch_delayed_robot_pick()
    app.robot.poll.return_value = {"status": "done"}

    app._robot_poll()

    assert app._automation_phase == "đang nhận diện"
    app._automation_schedule.assert_called_once()


def test_timeout_unblocks_with_warning(app):
    app._after_result({"conclusion": "FAIL", "items": [{"source": "yolo", "class_id": 0}]}, "a.png")
    app._dispatch_delayed_robot_pick()
    app._robot_job["deadline"] = 0.0
    app.robot.poll.return_value = None

    app._robot_poll()

    assert app._robot_job is None
    from src.ui import tkinter_app
    tkinter_app.messagebox.showwarning.assert_called_once()


def test_ready_state_accepts_teaching_controls(app):
    app._robot_state = "ready"
    app._robot_control_job = None
    app.robot.send_control.return_value = "control-1"

    app._send_robot_control("read_pose", {}, Mock())

    app.robot.send_control.assert_called_once_with("read_pose", {})


def test_learning_current_pose_updates_the_teaching_fields(app):
    app._robot_state = "ready"
    app._robot_control_job = None
    app.teach_point = SimpleNamespace(get=lambda: "B11")
    app._set_current_robot_pose = Mock()
    app._set_teach_pose = Mock()
    app._persist_teach_pose = Mock()

    TkAOIApp._save_current_robot_pose(app)
    callback = app._robot_control_job["on_done"]
    callback([1, 2, 3, 4])

    app._set_current_robot_pose.assert_called_once_with([1, 2, 3, 4])
    app._set_teach_pose.assert_called_once_with([1, 2, 3, 4])
    app._persist_teach_pose.assert_called_once_with("B11", [1, 2, 3, 4])


def test_stale_robot_heartbeat_transitions_offline(app, monkeypatch):
    app._robot_process = None
    app._robot_state = "ready"
    app.robot.status.return_value = {"status": "ready", "updated": 0}
    monkeypatch.setattr("src.ui.tkinter_app.time.time", lambda: 10)

    app._poll_robot_status()

    assert app._robot_state == "offline"
    app.root.after.assert_called_once_with(500, app._poll_robot_status)


def test_automation_waits_for_delayed_pick(app):
    app._automation_enabled = True
    app._automation_recaptures = 1
    app._automation_phase = "đang nhận diện"
    app._automation_schedule = Mock()

    app._automation_after_result({"conclusion": "FAIL", "items": [{"source": "yolo", "class_id": 0}]})

    assert app._automation_phase == "chờ robot gắp"
    assert app._automation_recaptures == 0
    app._automation_schedule.assert_called_once()


def test_automation_advances_pass_stack_after_each_successful_pick(app):
    app._automation_enabled = True
    app._pass_slot_index = 0
    app._after_result({"conclusion": "KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "items": []}, "GigE camera:B11")

    assert app._robot_delay_job["bin"] == "A11"
    app._dispatch_delayed_robot_pick()
    app.robot.poll.return_value = {"status": "done"}
    app._robot_poll()

    app._after_result({"conclusion": "KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "items": []}, "GigE camera:B11")
    assert app._robot_delay_job["bin"] == "A12"


def test_automation_stops_when_pass_stack_is_full(app):
    app._automation_enabled = True
    app._pass_slot_index = 3
    app._stop_automation = Mock()

    app._after_result({"conclusion": "KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "items": []}, "GigE camera:B11")

    assert app._robot_delay_job is None
    app._stop_automation.assert_called_once()


def test_shutdown_uses_emergency_stop(app):
    app = TkAOIApp.__new__(TkAOIApp)
    app._stop_automation = Mock()
    app._cam_disconnect = Mock()
    app._emergency_stop = Mock()
    app.root = Mock()

    app.shutdown()

    app._emergency_stop.assert_called_once()


def test_automation_recaptures_twice_then_pauses(app):
    app._automation_enabled = True
    app._automation_recaptures = 0
    app._automation_schedule = Mock()
    app._stop_automation = Mock()

    for _ in range(2):
        app._automation_after_result({"conclusion": "RECAPTURE"})

    assert app._automation_recaptures == 2
    assert app._automation_schedule.call_count == 2
    app._automation_after_result({"conclusion": "REWORK"})
    app._stop_automation.assert_called_once()


def test_without_robot_nothing_happens():
    app = TkAOIApp.__new__(TkAOIApp)
    app._after_result({"conclusion": "FAIL"}, "a.png")
    assert not app._robot_blocking()


def test_board_mode_selects_branch_b():
    assert _use_branch_b(SimpleNamespace())
    assert _use_branch_b(SimpleNamespace(board_mode=SimpleNamespace(get=lambda: "A")))
    assert not _use_branch_b(SimpleNamespace(board_mode=SimpleNamespace(get=lambda: "OTHER")))


def test_constructor_schedules_robot_autostart(monkeypatch, tmp_path):
    class Root:
        def __init__(self):
            self.after_calls = []

        def title(self, _text): pass
        def geometry(self, _text): pass
        def configure(self, **_kwargs): pass
        def minsize(self, *_args): pass
        def protocol(self, *_args): pass
        def after(self, delay, callback):
            self.after_calls.append((delay, callback))

    class Thread:
        def __init__(self, target, daemon):
            self.target, self.daemon = target, daemon

        def start(self): pass

    monkeypatch.setattr(TkAOIApp, "_build_ui", lambda self: None)
    monkeypatch.setattr(TkAOIApp, "_wire_shortcuts", lambda self: None)
    monkeypatch.setattr("src.ui.tkinter_app.threading.Thread", Thread)
    root = Root()

    app = TkAOIApp(root, weights_path=Path("weights.pt"), robot_dir=tmp_path,
                   robot_autostart="sim")

    assert app.robot_autostart == "sim"
    assert [delay for delay, _ in root.after_calls] == [400, 500, 750]
