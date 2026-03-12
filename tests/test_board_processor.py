from __future__ import annotations

from floptician.board_processor import BoardProcessor
from floptician.models import (
    AppConfig,
    BoardState,
    CommunityCard,
    TransitionState,
)
from tests.conftest import FakeDetector


def _make_config(app_config: AppConfig) -> AppConfig:
    """Ensure board_processor config has test-friendly thresholds."""
    app_config.board_processor.min_frames_to_show = 2
    app_config.board_processor.min_ms_to_show = 0  # immediate for tests
    app_config.board_processor.min_frames_to_remove = 2
    app_config.board_processor.min_ms_to_remove = 0
    return app_config


class TestBoardProcessorInitialState:
    def test_initial_state_is_not_showing(self, app_config: AppConfig):
        detector = FakeDetector()
        bp = BoardProcessor(_make_config(app_config), detector=detector)
        assert bp.board_state == BoardState.NOT_SHOWING
        assert bp.transition_state == TransitionState.NONE

    def test_empty_frame_stays_not_showing(self, app_config: AppConfig, blank_frame):
        detector = FakeDetector(detections=[])
        bp = BoardProcessor(_make_config(app_config), detector=detector)
        result = bp.process_frame(blank_frame)
        assert result.state == BoardState.NOT_SHOWING


class TestBoardProcessorTransitions:
    def _make_bp(self, app_config, detections):
        detector = FakeDetector(detections=detections)
        return BoardProcessor(_make_config(app_config), detector=detector)

    def test_cards_detected_triggers_appearing(self, app_config, sample_detections, blank_frame):
        bp = self._make_bp(app_config, sample_detections)

        bp.process_frame(blank_frame)
        # After first frame with detections, should be in APPEARING transition
        assert bp.board_state == BoardState.NOT_SHOWING
        assert bp.transition_state == TransitionState.APPEARING

    def test_consistent_detection_leads_to_showing(self, app_config, sample_detections, blank_frame):
        bp = self._make_bp(app_config, sample_detections)

        # Process enough frames for threshold (min_frames_to_show=2, min_ms_to_show=0)
        bp.process_frame(blank_frame)
        result = bp.process_frame(blank_frame)

        assert result.state == BoardState.SHOWING
        assert len(result.board) == 3

    def test_cards_disappear_triggers_disappearing(self, app_config, sample_detections, blank_frame):
        bp = self._make_bp(app_config, sample_detections)

        # Get to SHOWING state
        bp.process_frame(blank_frame)
        bp.process_frame(blank_frame)
        assert bp.board_state == BoardState.SHOWING

        # Now switch to no detections
        bp.yolo_processor = FakeDetector(detections=[])
        bp.process_frame(blank_frame)
        assert bp.transition_state == TransitionState.DISAPPEARING

    def test_removal_thresholds_clear_board(self, app_config, sample_detections, blank_frame):
        bp = self._make_bp(app_config, sample_detections)

        # Get to SHOWING state
        bp.process_frame(blank_frame)
        bp.process_frame(blank_frame)
        assert bp.board_state == BoardState.SHOWING

        # Process enough empty frames to trigger removal
        bp.yolo_processor = FakeDetector(detections=[])
        bp.process_frame(blank_frame)
        result = bp.process_frame(blank_frame)

        assert result.state == BoardState.NOT_SHOWING
        assert result.board == []


class TestBoardsMatch:
    def test_matching_boards(self, app_config):
        detector = FakeDetector()
        bp = BoardProcessor(_make_config(app_config), detector=detector)

        board1 = [
            CommunityCard(card="Ah", x=1, y=1),
            CommunityCard(card="Kd", x=2, y=1),
        ]
        board2 = [
            CommunityCard(card="Kd", x=3, y=1),
            CommunityCard(card="Ah", x=4, y=1),
        ]
        assert bp._boards_match(board1, board2) is True

    def test_non_matching_boards(self, app_config):
        detector = FakeDetector()
        bp = BoardProcessor(_make_config(app_config), detector=detector)

        board1 = [CommunityCard(card="Ah", x=1, y=1)]
        board2 = [CommunityCard(card="Kd", x=1, y=1)]
        assert bp._boards_match(board1, board2) is False

    def test_empty_boards_match(self, app_config):
        detector = FakeDetector()
        bp = BoardProcessor(_make_config(app_config), detector=detector)
        assert bp._boards_match([], []) is True


class TestBoardHasNewCards:
    def test_detects_new_cards(self, app_config):
        detector = FakeDetector()
        bp = BoardProcessor(_make_config(app_config), detector=detector)

        board1 = [
            CommunityCard(card="Ah", x=1, y=1),
            CommunityCard(card="Kd", x=2, y=1),
            CommunityCard(card="Qs", x=3, y=1),
        ]
        board2 = [
            CommunityCard(card="Ah", x=1, y=1),
            CommunityCard(card="Kd", x=2, y=1),
        ]
        assert bp._board_has_new_cards(board1, board2) is True

    def test_no_new_cards(self, app_config):
        detector = FakeDetector()
        bp = BoardProcessor(_make_config(app_config), detector=detector)

        board1 = [CommunityCard(card="Ah", x=1, y=1)]
        board2 = [CommunityCard(card="Ah", x=1, y=1)]
        assert bp._board_has_new_cards(board1, board2) is False
