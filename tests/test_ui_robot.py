from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.ui.tkinter_app import TkAOIApp, _use_branch_b


@pytest.fixture
def app(monkeypatch):
    app = TkAOIApp.__new__(TkAOIApp)
    app.robot = Mock()
    app.robot.send.return_value = "job-1"
    app._robot_job = None
    app._robot_review = None
    app.root = Mock()
    app.op_frame = Mock()
    app.lbl_robot = Mock()
    app._set_status = Mock()
    monkeypatch.setattr("src.ui.tkinter_app.messagebox.showwarning", Mock())
    monkeypatch.setattr("src.ui.tkinter_app.messagebox.showerror", Mock())
    return app


@pytest.mark.parametrize(("conclusion", "bin_name"), [
    ("FAIL", "FAIL"), ("KHÔNG THẤY LỖI (chưa hiệu chuẩn)", "PASS")])
def test_auto_pick_sends_one_command_and_blocks(app, conclusion, bin_name):
    app._after_result({"conclusion": conclusion}, "a.png")
    app.robot.send.assert_called_once_with(bin_name, conclusion, "auto", "a.png")
    assert app._robot_job["id"] == "job-1"
    assert app._robot_blocking()


def test_review_waits_for_operator(app):
    app._after_result({"conclusion": "REVIEW"}, "a.png")
    app.robot.send.assert_not_called()
    app.op_frame.pack.assert_called_once()
    assert app._robot_blocking()
    app._operator_choice("PASS")
    app.op_frame.pack_forget.assert_called_once()
    app.robot.send.assert_called_once_with("PASS", "REVIEW", "operator", "a.png")


def test_operator_can_skip_pick(app):
    app._after_result({"conclusion": "REVIEW"}, "a.png")
    app._operator_choice(None)
    app.robot.send.assert_not_called()
    assert not app._robot_blocking()


@pytest.mark.parametrize("conclusion", ["RECAPTURE", "CONFIRM"])
def test_capture_actions_never_pick(app, conclusion):
    app._after_result({"conclusion": conclusion}, "a.png")
    app.robot.send.assert_not_called()
    assert not app._robot_blocking()


def test_ack_unblocks(app):
    app._after_result({"conclusion": "FAIL"}, "a.png")
    app.robot.poll.return_value = {"status": "done"}
    app._robot_poll()
    assert app._robot_job is None and not app._robot_blocking()


def test_timeout_unblocks_with_warning(app, monkeypatch):
    app._after_result({"conclusion": "FAIL"}, "a.png")
    app._robot_job["deadline"] = 0.0
    app.robot.poll.return_value = None
    app._robot_poll()
    assert app._robot_job is None
    from src.ui import tkinter_app
    tkinter_app.messagebox.showwarning.assert_called_once()


def test_without_robot_nothing_happens():
    app = TkAOIApp.__new__(TkAOIApp)
    app._after_result({"conclusion": "FAIL"}, "a.png")   # no robot attribute: no error
    assert not app._robot_blocking()


def test_board_mode_selects_branch_b():
    assert _use_branch_b(SimpleNamespace())
    assert _use_branch_b(SimpleNamespace(board_mode=SimpleNamespace(get=lambda: "A")))
    assert not _use_branch_b(SimpleNamespace(board_mode=SimpleNamespace(get=lambda: "OTHER")))
