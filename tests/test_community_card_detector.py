from __future__ import annotations

import pytest

from floptician.community_card_detector import CommunityCardDetector
from floptician.models import BoardConfiguration, BoardProcessorConfig, BoundingBox, CardDetection, CommunityCard


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


def _make_three_row(
    top_cards: list[str],
    mid_cards: list[str],
    bot_cards: list[str],
    top_y: float = 100,
    mid_y: float = 200,
    bot_y: float = 300,
    card_height: float = 60,
    x_spacing: float = 100,
) -> list[CardDetection]:
    """Create a 3-row layout (top + middle + bottom)."""
    detections: list[CardDetection] = []
    for y_center, cards in ((top_y, top_cards), (mid_y, mid_cards), (bot_y, bot_cards)):
        y_start = y_center - card_height / 2
        y_end = y_center + card_height / 2
        for i, card in enumerate(cards):
            detections.append(
                CardDetection(
                    card=card,
                    confidence=0.90,
                    box=BoundingBox(100 + i * x_spacing, y_start, 160 + i * x_spacing, y_end),
                )
            )
    return detections


class TestDetectCommunityCards:
    def test_fewer_than_3_returns_empty(self, detector):
        detections = _make_row(["Ah", "Kd"])
        cards, config = detector.detect_community_cards(detections)
        assert cards == []
        assert config == BoardConfiguration.NO_BOARD

    def test_3_cards_single_row(self, detector):
        detections = _make_row(["Ah", "Kd", "Qs"])
        cards, config = detector.detect_community_cards(detections)
        assert len(cards) == 3
        assert all(isinstance(c, CommunityCard) for c in cards)
        assert {c.card for c in cards} == {"Ah", "Kd", "Qs"}
        assert config == BoardConfiguration.SINGLE_ROW

    def test_5_cards_single_row(self, detector):
        detections = _make_row(["Ah", "Kd", "Qs", "Jc", "Th"])
        cards, config = detector.detect_community_cards(detections)
        assert len(cards) == 5
        assert all(c.y == 1 for c in cards)
        assert config == BoardConfiguration.SINGLE_ROW

    def test_x_coordinates_are_sequential(self, detector):
        detections = _make_row(["Ah", "Kd", "Qs"])
        cards, _ = detector.detect_community_cards(detections)
        x_coords = sorted(c.x for c in cards)
        assert x_coords == [1, 2, 3]

    def test_two_rows_detected(self, detector):
        top_row = _make_row(["Ah", "Kd", "Qs"], y_start=100, y_end=180)
        bottom_row = _make_row(["2c", "3d", "4h"], y_start=300, y_end=380)
        detections = top_row + bottom_row
        cards, config = detector.detect_community_cards(detections)
        assert len(cards) == 6
        y_values = {c.y for c in cards}
        assert y_values == {1, 3}
        assert config == BoardConfiguration.TWO_ROWS


class TestRunItTwiceFlop:
    def test_3_plus_2_plus_2(self, detector):
        detections = _make_three_row(
            top_cards=["Ah", "Kd"],
            mid_cards=["Qs", "Jc", "Th"],
            bot_cards=["9s", "8h"],
        )
        cards, config = detector.detect_community_cards(detections)
        assert config == BoardConfiguration.RUN_IT_TWICE_FLOP
        assert len(cards) == 7
        top = [c for c in cards if c.y == 1]
        mid = [c for c in cards if c.y == 2]
        bot = [c for c in cards if c.y == 3]
        assert len(top) == 2
        assert len(mid) == 3
        assert len(bot) == 2

    def test_3_plus_1_plus_1(self, detector):
        detections = _make_three_row(
            top_cards=["Ah"],
            mid_cards=["Qs", "Jc", "Th"],
            bot_cards=["9s"],
        )
        cards, config = detector.detect_community_cards(detections)
        assert config == BoardConfiguration.RUN_IT_TWICE_FLOP
        assert len(cards) == 5

    def test_unequal_runout_rows_no_match(self, detector):
        """Top and bottom runout rows must have equal card counts."""
        detections = _make_three_row(
            top_cards=["Ah", "Kd"],
            mid_cards=["Qs", "Jc", "Th"],
            bot_cards=["9s"],
        )
        _cards, config = detector.detect_community_cards(detections)
        # Should NOT be RUN_IT_TWICE_FLOP since 2 != 1
        assert config != BoardConfiguration.RUN_IT_TWICE_FLOP


class TestRunItTwiceTurn:
    def test_4_plus_1_plus_1(self, detector):
        detections = _make_three_row(
            top_cards=["Ah"],
            mid_cards=["Qs", "Jc", "Th", "9s"],
            bot_cards=["8h"],
        )
        cards, config = detector.detect_community_cards(detections)
        assert config == BoardConfiguration.RUN_IT_TWICE_TURN
        assert len(cards) == 6
        top = [c for c in cards if c.y == 1]
        mid = [c for c in cards if c.y == 2]
        bot = [c for c in cards if c.y == 3]
        assert len(top) == 1
        assert len(mid) == 4
        assert len(bot) == 1


class TestStrayCardRejection:
    def test_single_stray_card_no_main_row(self, detector):
        """A lone card far from any main row should not create a false positive."""
        detections = [
            CardDetection(card="Ah", confidence=0.90, box=BoundingBox(100, 100, 160, 160)),
        ]
        cards, config = detector.detect_community_cards(detections)
        assert cards == []
        assert config == BoardConfiguration.NO_BOARD


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

    def test_valid_run_it_twice_flop(self, detector):
        cards = [
            CommunityCard(card="Ah", x=1, y=1),
            CommunityCard(card="Kd", x=2, y=1),
            CommunityCard(card="Qs", x=1, y=2),
            CommunityCard(card="Jc", x=2, y=2),
            CommunityCard(card="Th", x=3, y=2),
            CommunityCard(card="9s", x=1, y=3),
            CommunityCard(card="8h", x=2, y=3),
        ]
        assert detector.validate_configuration(cards) is True

    def test_valid_run_it_twice_turn(self, detector):
        cards = [
            CommunityCard(card="Ah", x=1, y=1),
            CommunityCard(card="Qs", x=1, y=2),
            CommunityCard(card="Jc", x=2, y=2),
            CommunityCard(card="Th", x=3, y=2),
            CommunityCard(card="9s", x=4, y=2),
            CommunityCard(card="8h", x=1, y=3),
        ]
        assert detector.validate_configuration(cards) is True
