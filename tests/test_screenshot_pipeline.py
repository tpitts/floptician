"""Screenshot-based full-pipeline tests.

Auto-discovers tests/screenshots/**/*.expected.json, pairs with .png,
and runs the full YOLO → CommunityCardDetector pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from floptician.board_processor import BoardProcessor
from floptician.community_card_detector import CommunityCardDetector
from floptician.models import AppConfig, BoardConfiguration, BoardProcessorConfig, BoardState, YOLOConfig
from floptician.yolo_processor import YOLOProcessor
from tests.conftest import FakeDetector

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8l-2026-03-10.pt"


def _discover_test_cases() -> list[tuple[str, Path, Path]]:
    """Return (test_id, png_path, json_path) tuples for parametrization."""
    cases = []
    for json_path in sorted(SCREENSHOTS_DIR.rglob("*.expected.json")):
        png_path = json_path.with_name(json_path.name.replace(".expected.json", ".png"))
        if not png_path.exists():
            raise FileNotFoundError(f"Missing screenshot for {json_path}: {png_path}")
        rel = png_path.relative_to(SCREENSHOTS_DIR)
        test_id = str(rel).replace("\\", "/").replace(".png", "")
        cases.append((test_id, png_path, json_path))
    return cases


TEST_CASES = _discover_test_cases()


@pytest.fixture(scope="session")
def yolo_processor():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    config = YOLOConfig(model=str(MODEL_PATH))
    return YOLOProcessor(config)


@pytest.fixture(scope="session")
def detector():
    config = BoardProcessorConfig(vertical_alignment_threshold=0.20)
    return CommunityCardDetector(config)


@pytest.mark.screenshot
@pytest.mark.parametrize(
    "test_id,png_path,json_path",
    TEST_CASES,
    ids=[c[0] for c in TEST_CASES],
)
def test_screenshot_detection(test_id, png_path, json_path, yolo_processor, detector):
    # Load expected
    with open(json_path) as f:
        expected = json.load(f)

    expected_config = BoardConfiguration[expected["configuration"]]
    expected_cards = {(c["card"], c["x"], c["y"]) for c in expected["cards"]}

    # Run pipeline
    frame = cv2.imread(str(png_path))
    assert frame is not None, f"Failed to read image: {png_path}"

    detections = yolo_processor.process_frame(frame)
    community_cards, configuration = detector.detect_community_cards(detections)

    actual_cards = {(c.card, c.x, c.y) for c in community_cards}

    # Assert
    assert configuration == expected_config, (
        f"Configuration mismatch: expected {expected_config.name}, got {configuration.name}"
    )
    assert actual_cards == expected_cards, (
        f"Card set mismatch.\n  Expected: {sorted(expected_cards)}\n  Actual:   {sorted(actual_cards)}"
    )

    # Replay these real model detections through stabilization, then briefly occlude
    # them. Every supported screenshot layout must retain its complete snapshot.
    now = 0.0
    source = FakeDetector(detections)
    board_processor = BoardProcessor(AppConfig(), detector=source, clock=lambda: now)
    for observation_time in (0.0, 0.6):
        now = observation_time
        pending = board_processor.process_frame(frame)
        assert pending.board == []
        assert pending.configuration == BoardConfiguration.NO_BOARD
    now = 1.25
    confirmed = board_processor.process_frame(frame)
    assert confirmed.state == BoardState.SHOWING
    assert confirmed.configuration == expected_config
    assert {(c.card, c.x, c.y) for c in confirmed.board} == expected_cards
    source._detections = []
    now = 2.0
    occluded = board_processor.process_frame(frame)
    assert occluded.configuration == confirmed.configuration
    assert occluded.board == confirmed.board
