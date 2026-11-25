from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from src.ui.tkinter_app import TkAOIApp


@pytest.fixture
def immediate_worker(monkeypatch):
    def immediate_thread(*, target, daemon):
        return SimpleNamespace(start=target)

    monkeypatch.setattr('src.ui.tkinter_app.threading.Thread', immediate_thread)


def test_error_replaces_old_pass_and_disables_stale_saved_frame(monkeypatch):
    app = TkAOIApp.__new__(TkAOIApp)
    for name in ('banner', 'lbl_verdict', 'lbl_verdict_sub', 'lbl_metrics',
                 'tree', 'lbl_defects', 'img_annotated', 'btn_save_inspection',
                 '_set_status'):
        setattr(app, name, Mock())
    app.tree.get_children.return_value = ('old-row',)
    app.img_annotated._frame_bgr = object()
    app._last_inspection_frame = object()
    app._pending_camera_regions = ('old',)
    monkeypatch.setattr('src.ui.tkinter_app.messagebox.showerror', Mock())
    TkAOIApp._inspect_error(app, 'camera lost')
    assert app.lbl_verdict.config.call_args.kwargs['text'] == '— KHÔNG KIỂM TRA ĐƯỢC'
    assert app.lbl_verdict_sub.config.call_args.kwargs['text'] == 'camera lost'
    app.tree.delete.assert_called_once_with('old-row')
    app.img_annotated.config.assert_called_once_with(image='')
    assert app.img_annotated._frame_bgr is None
    app.btn_save_inspection.config.assert_called_once_with(state='disabled')
    assert app._last_inspection_frame is None
    assert app._pending_camera_regions is None


@pytest.mark.parametrize("weights_path", [None, "real_all_boards.pt"])
def test_dispatch_error_route_calls_inspect_error(immediate_worker, weights_path):
    """Duong (a): engine nem RuntimeError trong _run_inspect_worker
    -> _inspect_error duoc goi (khong qua _show_result 'inspection_unavailable')."""
    app = SimpleNamespace(_busy=False, _set_busy=Mock(), _inspect_error=Mock(),
                          _pending_camera_regions=('old',), weights_path=weights_path,
                          _real_ab_context=0, _real_ab_source="GigE camera")
    app.engine = Mock()
    app.real_ab = Mock()
    inspector = app.real_ab if weights_path else app.engine
    inspector.inspect.side_effect = RuntimeError("bo loi")
    app.root = Mock()
    app.root.after.side_effect = lambda ms, cb: cb()

    TkAOIApp._run_inspect_worker(app, img=np.zeros((64, 64, 3), np.uint8),
                                 pinned=None, source="GigE camera")

    app._inspect_error.assert_called_once()
    inspector.inspect.assert_called_once()
    assert "bo loi" in str(app._inspect_error.call_args)
    assert app._set_busy.call_args_list == [call(True), call(False)]


def test_file_confirm_reuses_pending_only_for_same_image(immediate_worker):
    app = TkAOIApp.__new__(TkAOIApp)
    app._busy = False
    app._grabbing_cam = False
    app._set_busy = Mock()
    app._show_result = Mock()
    app.nb = Mock()
    app.weights_path = "real_all_boards.pt"
    app._real_ab_context = 0
    app._real_ab_source = None
    app._pending_camera_regions = None
    app.root = Mock()
    app.root.after.side_effect = lambda ms, cb: cb()
    app.real_ab = Mock()
    candidate = object()
    app.real_ab.inspect.return_value = {"b_pending": candidate}
    frame = np.zeros((64, 64, 3), np.uint8)
    for image in (frame, frame, frame.copy()):
        app._run_inspect_worker(image, pinned=None, source="test.png")

    pending = [args.kwargs["pending"] for args in app.real_ab.inspect.call_args_list]
    assert pending == [None, candidate, None]


def test_real_ab_discards_capture_after_source_changes():
    app = SimpleNamespace(root=Mock(), cam_mgr=Mock(), _real_ab_context=0,
                          _finish_calibrated_capture=Mock(), _run_inspect=Mock())

    def grab(**kwargs):
        app._real_ab_context = 1
        return np.zeros((64, 64, 3), np.uint8)

    app.cam_mgr.grab_frame_bgr.side_effect = grab
    app.root.after.side_effect = lambda ms, cb: cb()

    TkAOIApp._inspect_camera(app, context=0)

    app._finish_calibrated_capture.assert_called_once()
    app._run_inspect.assert_not_called()
