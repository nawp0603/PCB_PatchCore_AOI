import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.calibrated_camera import CalibratedCameraInspection


def test_explicit_fold_profile_loads_its_own_threshold(tmp_path, monkeypatch):
    profile = tmp_path / 'fold'
    profile.mkdir()
    (profile / 'settings.json').write_text(json.dumps({'threshold': 5.4}))
    for name in ('template_A.png', 'template_mask.png', 'template_provenance.png'):
        cv2.imwrite(str(profile / name), np.full((64, 64, 3), 255, np.uint8))
    np.save(profile / 'residual_tolerance.npy', np.full((64, 64), 8., np.float32))
    monkeypatch.setattr(CalibratedCameraInspection, '_features', lambda *args: (None, None))
    engine = CalibratedCameraInspection(tmp_path, profile_dir=profile)
    assert engine.threshold == 5.4
    assert np.all(engine.quality_tolerance == 8)
    assert np.all(engine.detection_tolerance == 4)


def test_alignment_failure_exposes_no_transform_and_preserves_legacy_pair(monkeypatch):
    engine = object.__new__(CalibratedCameraInspection)
    engine.threshold = 6.
    monkeypatch.setattr(engine, '_align', lambda frame: None)
    frame = np.zeros((64, 64, 3), np.uint8)
    result, aligned, transform = engine.inspect_with_transform(frame)
    assert result.action == 'RECAPTURE'
    assert aligned is frame
    assert transform is None
    old_result, old_aligned = engine.inspect(frame)
    assert old_result == result
    assert old_aligned is frame


def test_transform_is_same_one_used_to_warp_image(monkeypatch):
    engine = object.__new__(CalibratedCameraInspection)
    engine.threshold = 6.
    engine.gray = np.zeros((64, 64), np.uint8)
    engine.mask = np.ones((64, 64), bool)
    transform = np.array([[1., 0., 20.], [0., 1., 0.], [0., 0., 1.]])
    monkeypatch.setattr(engine, '_align', lambda frame: transform)
    frame = np.zeros((64, 64, 3), np.uint8)
    frame[20:30, 10:20] = 255
    result, aligned, actual = engine.inspect_with_transform(frame)
    assert result.action == 'RECAPTURE'  # shifted image does not cover 98% of scope
    np.testing.assert_array_equal(actual, transform)
    np.testing.assert_array_equal(aligned, cv2.warpPerspective(frame, transform, (64, 64)))
