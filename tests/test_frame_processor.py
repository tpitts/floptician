from __future__ import annotations

import numpy as np

from floptician.models import BoardResult, BoardState, CommunityCard
from tests.conftest import FakeWebSocketServer


class TestIsValidFrame:
    """Test frame validation without instantiating FrameProcessor (test the logic)."""

    def _is_valid_frame(self, frame: np.ndarray, previous_frame: np.ndarray | None = None) -> bool:
        """Reimplementation of FrameProcessor.is_valid_frame for unit testing."""
        if frame is None:
            return False

        total_pixels = frame.shape[0] * frame.shape[1]
        black_pixels = np.sum(frame == 0) / 3
        white_pixels = np.sum(frame == 255) / 3

        if black_pixels / total_pixels > 0.5:
            return False

        if white_pixels / total_pixels > 0.5:
            return False

        return not (previous_frame is not None and np.array_equal(frame, previous_frame))

    def test_black_frame_rejected(self, blank_frame):
        assert self._is_valid_frame(blank_frame) is False

    def test_white_frame_rejected(self):
        white = np.full((360, 640, 3), 255, dtype=np.uint8)
        assert self._is_valid_frame(white) is False

    def test_normal_frame_accepted(self, normal_frame):
        assert self._is_valid_frame(normal_frame) is True

    def test_frozen_frame_rejected(self, normal_frame):
        assert self._is_valid_frame(normal_frame, previous_frame=normal_frame) is False

    def test_different_frames_accepted(self, normal_frame):
        rng = np.random.default_rng(99)
        other_frame = rng.integers(30, 200, size=(360, 640, 3), dtype=np.uint8)
        assert self._is_valid_frame(other_frame, previous_frame=normal_frame) is True


class TestWebSocketBroadcast:
    def test_fake_ws_records_messages(self):
        ws = FakeWebSocketServer()
        ws.send_message({"state": "Showing", "board": []})
        assert len(ws.messages) == 1
        assert ws.messages[0]["state"] == "Showing"

    def test_board_result_serialization(self):
        result = BoardResult(
            timestamp=1234.0,
            state=BoardState.SHOWING,
            board=[CommunityCard(card="Ah", x=1, y=1, confidence=0.95)],
            debug_info={},
        )
        d = result.to_dict()
        assert d["state"] == "Showing"
        assert len(d["board"]) == 1
        assert d["board"][0]["card"] == "Ah"
