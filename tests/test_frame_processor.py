from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from floptician.exceptions import CameraError
from floptician.frame_processor import FrameProcessor
from floptician.models import FrameInfo
from tests.conftest import FakeCamera
from tests.test_board_stability import Replay, row, signature


@pytest.fixture
def pipeline(app_config, fake_obs, fake_ws, monkeypatch):
    # Terminal input is an external boundary; exercise the actual frame processor.
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(kbhit=lambda: False, getch=lambda: b""))
    replay = Replay(app_config)
    processor = FrameProcessor(app_config, fake_obs, replay.processor, fake_ws)
    app_config.capture.camera_manager = FakeCamera()
    return processor, replay


@pytest.mark.parametrize("kind", ["black", "white", "frozen", "none"])
def test_invalid_frame_breaks_confirmation(pipeline, normal_frame, kind):
    processor, replay = pipeline
    replay(0, row())
    replay(0.6, row())
    frames = {
        "black": np.zeros_like(normal_frame),
        "white": np.full_like(normal_frame, 255),
        "frozen": normal_frame,
        "none": None,
    }
    processor.previous_frame = normal_frame.copy()
    assert processor.process_frame(FrameInfo(1, frames[kind], 1)) is None
    assert replay(2, row()).board == []
    assert replay(2.6, row()).board == []
    assert len(replay(3.25, row()).board) == 3


def test_actual_frame_validation_uses_current_black_threshold(pipeline, normal_frame):
    processor, _ = pipeline
    frame = normal_frame.copy()
    frame[: int(frame.shape[0] * 0.75)] = 0
    assert processor.is_valid_frame(frame)
    assert not processor.is_valid_frame(frame.copy())


def test_failed_capture_breaks_confirmation(pipeline):
    processor, replay = pipeline
    replay(0, row())
    replay(0.6, row())
    assert processor.capture_frame() is None
    assert replay(2, row()).board == []
    assert replay(2.6, row()).board == []
    assert len(replay(3.25, row()).board) == 3


def test_capture_exception_breaks_confirmation(pipeline, monkeypatch):
    processor, replay = pipeline
    replay(0, row())
    replay(0.6, row())

    def fail():
        raise OSError("camera disconnected")

    monkeypatch.setattr(processor.config.capture.camera_manager, "get_frame", fail)
    with pytest.raises(CameraError):
        processor.capture_frame()
    assert replay(2, row()).board == []
    assert replay(2.6, row()).board == []


def test_inference_failure_produces_no_broadcastable_result(pipeline, normal_frame, monkeypatch):
    processor, replay = pipeline
    original = replay.confirm()

    def fail(_):
        raise RuntimeError("inference failed")

    with monkeypatch.context() as m:
        m.setattr(replay.detector, "process_frame", fail)
        assert processor.process_frame(FrameInfo(1, normal_frame, 2)) is None
        assert processor.frame_count == 0
    assert signature(replay(3, row())) == signature(original)


def test_capture_loop_only_broadcasts_successful_results(pipeline, normal_frame, monkeypatch):
    processor, replay = pipeline
    original = replay.confirm()
    readings = iter([normal_frame, normal_frame + 1, normal_frame + 2])
    calls = 0

    def capture():
        nonlocal calls
        try:
            frame = next(readings)
        except StopIteration:
            processor.running = False
            return False, None
        calls += 1
        replay.now = 2 + calls
        return True, frame

    def detect(_):
        if calls == 2:
            raise RuntimeError("transient failure")
        return row()

    monkeypatch.setattr(processor.config.capture.camera_manager, "get_frame", capture)
    monkeypatch.setattr(replay.detector, "process_frame", detect)
    monkeypatch.setattr(processor, "check_for_quit", lambda: False)
    monkeypatch.setattr("floptician.frame_processor.time.sleep", lambda _: None)
    processor.run()
    messages = processor.websocket_server.messages
    assert len(messages) == 2
    assert [m["frame_id"] for m in messages] == [1, 3]
    assert all(m["configuration"] == original.configuration.name for m in messages)
    assert all(len(m["board"]) == 3 for m in messages)
    assert processor.config.capture.camera_manager.released
