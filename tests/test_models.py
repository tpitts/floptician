from __future__ import annotations

from floptician.models import (
    AppConfig,
    BoardConfiguration,
    BoardResult,
    BoardState,
    BoundingBox,
    CaptureMode,
    CardDetection,
    CommunityCard,
    TransitionState,
)


class TestEnums:
    def test_board_state_values(self):
        assert BoardState.NOT_SHOWING.value == "Not Showing"
        assert BoardState.SHOWING.value == "Showing"

    def test_transition_state_values(self):
        assert TransitionState.NONE.value == "None"
        assert TransitionState.APPEARING.value == "Appearing"
        assert TransitionState.DISAPPEARING.value == "Disappearing"

    def test_capture_mode_values(self):
        assert CaptureMode.OBS_WEBSOCKET.value == "obs_websocket"
        assert CaptureMode.DIRECT_WEBCAM.value == "direct_webcam"

    def test_board_configuration_members(self):
        assert BoardConfiguration.NO_BOARD is not None
        assert BoardConfiguration.SINGLE_ROW is not None
        assert BoardConfiguration.TWO_ROWS is not None
        assert BoardConfiguration.CHIHUAHUA is not None


class TestCardDetection:
    def test_creation(self):
        det = CardDetection(card="Ah", confidence=0.95, box=BoundingBox(0, 0, 50, 50))
        assert det.card == "Ah"
        assert det.confidence == 0.95
        assert det.box.width == 50

    def test_box_properties(self):
        det = CardDetection(card="Kd", confidence=0.90, box=BoundingBox(10, 20, 30, 40))
        assert det.box.center_x == 20.0
        assert det.box.center_y == 30.0


class TestCommunityCard:
    def test_creation(self):
        cc = CommunityCard(card="Ah", x=1, y=1, confidence=0.95)
        assert cc.card == "Ah"
        assert cc.x == 1
        assert cc.y == 1

    def test_default_confidence(self):
        cc = CommunityCard(card="Kd", x=2, y=1)
        assert cc.confidence == 1.0


class TestBoardResult:
    def test_to_dict(self):
        result = BoardResult(
            timestamp=100.0,
            state=BoardState.SHOWING,
            board=[CommunityCard(card="Ah", x=1, y=1, confidence=0.95)],
            debug_info={"frame_id": 1},
        )
        d = result.to_dict()
        assert d["timestamp"] == 100.0
        assert d["state"] == "Showing"
        assert len(d["board"]) == 1
        assert d["board"][0]["card"] == "Ah"
        assert d["debug_info"]["frame_id"] == 1

    def test_empty_board(self):
        result = BoardResult(
            timestamp=0.0,
            state=BoardState.NOT_SHOWING,
            board=[],
        )
        d = result.to_dict()
        assert d["board"] == []
        assert d["state"] == "Not Showing"


class TestAppConfig:
    def test_defaults(self):
        config = AppConfig()
        assert config.debug is False
        assert config.host == "localhost"
        assert config.http_port == 8000
        assert config.obs.port == 4455
        assert config.capture.fps == 1.8
        assert config.yolo.confidence_threshold == 0.70
