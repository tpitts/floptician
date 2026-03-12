from __future__ import annotations

import pytest

from floptician.community_card_detector import CommunityCardDetector
from floptician.models import BoardProcessorConfig, BoundingBox, CardDetection, CommunityCard


@pytest.fixture
def detector() -> CommunityCardDetector:
    config = BoardProcessorConfig(vertical_alignment_threshold=0.20)
    return CommunityCardDetector(config)


def _make_row(
    cards: list[str], y_start: float = 200, y_end: float = 280, x_spacing: float = 100
) -> list[CardDetection]:
    """Create a horizontal row of card detections."""
    return [
        CardDetection(
            card=card,
            confidence=0.90,
            box=BoundingBox(100 + i * x_spacing, y_start, 160 + i * x_spacing, y_end),
        )
        for i, card in enumerate(cards)
    ]


class TestDetectCommunityCards:
    def test_fewer_than_3_returns_empty(self, detector):
        detections = _make_row(["Ah", "Kd"])
        result = detector.detect_community_cards(detections)
        assert result == []

    def test_3_cards_single_row(self, detector):
        detections = _make_row(["Ah", "Kd", "Qs"])
        result = detector.detect_community_cards(detections)
        assert len(result) == 3
        assert all(isinstance(c, CommunityCard) for c in result)
        assert {c.card for c in result} == {"Ah", "Kd", "Qs"}

    def test_5_cards_single_row(self, detector):
        detections = _make_row(["Ah", "Kd", "Qs", "Jc", "Th"])
        result = detector.detect_community_cards(detections)
        assert len(result) == 5
        # All should be y=1 (single row)
        assert all(c.y == 1 for c in result)

    def test_x_coordinates_are_sequential(self, detector):
        detections = _make_row(["Ah", "Kd", "Qs"])
        result = detector.detect_community_cards(detections)
        x_coords = sorted(c.x for c in result)
        assert x_coords == [1, 2, 3]

    def test_two_rows_detected(self, detector):
        top_row = _make_row(["Ah", "Kd", "Qs"], y_start=100, y_end=180)
        bottom_row = _make_row(["2c", "3d", "4h"], y_start=300, y_end=380)
        detections = top_row + bottom_row
        result = detector.detect_community_cards(detections)
        assert len(result) == 6
        y_values = {c.y for c in result}
        assert y_values == {1, 3}


class TestValidateConfiguration:
    def test_valid_single_row_3(self, detector):
        cards = [
            CommunityCard(card="Ah", x=1, y=1),
            CommunityCard(card="Kd", x=2, y=1),
            CommunityCard(card="Qs", x=3, y=1),
        ]
        assert detector.validate_configuration(cards) is True

    def test_valid_single_row_5(self, detector):
        cards = [CommunityCard(card=f"{i}h", x=i, y=1) for i in range(1, 6)]
        assert detector.validate_configuration(cards) is True

    def test_invalid_single_row_2(self, detector):
        cards = [
            CommunityCard(card="Ah", x=1, y=1),
            CommunityCard(card="Kd", x=2, y=1),
        ]
        assert detector.validate_configuration(cards) is False
