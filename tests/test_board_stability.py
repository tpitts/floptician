from __future__ import annotations

from dataclasses import replace

import pytest

from floptician.board_processor import BoardProcessor
from floptician.models import BoardConfiguration, BoardState, BoundingBox, CardDetection
from tests.conftest import FakeDetector


def row(names="Ah Kd Qs", y=200):
    return [
        CardDetection(name, 0.95, BoundingBox(100 + i * 100, y, 160 + i * 100, y + 80))
        for i, name in enumerate(names.split())
    ]


def signature(result):
    return result.configuration, tuple((c.card, c.x, c.y) for c in result.board)


class Replay:
    """Exercise real grouping and stabilization without a camera or sleeps."""

    def __init__(self, config):
        self.now = 0.0
        self.detector = FakeDetector()
        self.processor = BoardProcessor(config, detector=self.detector, clock=lambda: self.now)

    def __call__(self, now, detections):
        self.now = now
        self.detector._detections = detections
        return self.processor.process_frame(None)

    def confirm(self, detections=None, start=0):
        detections = row() if detections is None else detections
        self(start, detections)
        self(start + 0.6, detections)
        return self(start + 1.25, detections)


@pytest.fixture
def replay(app_config):
    return Replay(app_config)


def test_initial_empty_board(replay):
    result = replay(0, [])
    assert result.state == BoardState.NOT_SHOWING
    assert signature(result) == (BoardConfiguration.NO_BOARD, ())


def test_appearance_requires_both_count_and_elapsed_time(replay):
    for t in [0, 0.1, 0.2, 1.199]:
        assert replay(t, row()).board == []
    assert len(replay(1.2, row()).board) == 3


def test_elapsed_time_alone_cannot_confirm(replay):
    replay(0, row())
    assert replay(1.2, row()).board == []
    assert len(replay(1.3, row()).board) == 3


@pytest.mark.parametrize(
    "replacement",
    [
        row("Ah Kd Qs Jc"),  # addition
        row("Ah Kd Js"),  # correction
        row("Kd Ah Qs"),  # rearrangement
        row() + row("Jc", y=100) + row("Th", y=300),  # run-it-twice layout
    ],
)
def test_all_replacements_use_appearance_gate_without_clearing(replay, replacement):
    original = replay.confirm()
    for t in [2, 2.6, 3.19]:
        assert signature(replay(t, replacement)) == signature(original)
    updated = replay(3.25, replacement)
    assert updated.state == BoardState.SHOWING
    assert {c.card for c in updated.board} == {d.card for d in replacement}
    assert signature(updated) != signature(original)
    assert signature(replay(4, replacement)) == signature(updated)


def test_configuration_is_part_of_confirmation_even_without_position_change(replay, monkeypatch):
    original = replay.confirm()
    monkeypatch.setattr(
        replay.processor.community_card_detector,
        "detect_community_cards",
        lambda _: (original.board, BoardConfiguration.TWO_ROWS),
    )
    assert signature(replay(2, row())) == signature(original)
    assert signature(replay(2.6, row())) == signature(original)
    assert replay(3.25, row()).configuration == BoardConfiguration.TWO_ROWS


def test_new_candidate_restarts_its_own_timer(replay):
    replay(0, row())
    replay(0.6, row())
    different = row("2h 3d 4s")
    replay(1, different)
    replay(1.5, different)
    assert replay(2, different).board == []
    assert len(replay(2.25, different).board) == 3


def test_alternating_candidates_never_accumulate_agreement(replay):
    original = replay.confirm()
    for i in range(20):
        candidate = row("Ah Kd Qs Jc") if i % 2 else row("Ah Kd Qs Th")
        assert signature(replay(2 + i * 0.6, candidate)) == signature(original)


def test_return_to_confirmed_board_cancels_candidate(replay):
    original = replay.confirm()
    addition = row("Ah Kd Qs Jc")
    replay(2, addition)
    replay(2.6, addition)
    replay(3, row())
    replay(3.6, addition)
    assert signature(replay(4.2, addition)) == signature(original)
    assert len(replay(4.85, addition).board) == 4


def test_confidence_pixel_jitter_and_detection_order_do_not_restart_candidate(replay):
    detections = row()
    replay(0, detections)
    jittered = [replace(d, confidence=0.8, box=replace(d.box, x1=d.box.x1 + 2)) for d in detections]
    replay(0.6, list(reversed(jittered)))
    assert len(replay(1.25, detections).board) == 3


def test_single_miss_preserves_cards_positions_and_layout(replay):
    detections = row() + row("Jc", y=100) + row("Th", y=300)
    original = replay.confirm(detections)
    missed = replay(2, [])
    assert missed.state == BoardState.SHOWING
    assert signature(missed) == signature(original)
    assert signature(replay(2.6, detections)) == signature(original)


def test_subset_is_occlusion_not_a_fast_replacement(replay):
    original = replay.confirm(row("Ah Kd Qs Jc Th"))
    for t in [2, 2.6, 3.2, 3.8, 4.4, 5, 5.6, 6.19]:
        assert signature(replay(t, row())) == signature(original)
    cleared = replay(6.25, row())
    assert cleared.state == BoardState.NOT_SHOWING
    assert signature(cleared) == (BoardConfiguration.NO_BOARD, ())
    assert replay(6.85, row()).board == []
    assert replay(7.45, row()).board == []
    assert len(replay(8.1, row()).board) == 3


def test_clearing_requires_both_count_and_elapsed_time(replay):
    original = replay.confirm()
    for i in range(6):
        assert signature(replay(2 + i * 0.1, [])) == signature(original)
    assert signature(replay(6.19, [])) == signature(original)
    assert replay(6.25, []).board == []


def test_clearing_waits_for_frame_count_even_after_time_threshold(replay):
    original = replay.confirm()
    for t in [2, 3, 4, 5, 6.25]:
        assert signature(replay(t, [])) == signature(original)
    assert replay(6.3, []).board == []


def test_different_missing_cards_do_not_restart_clearing(replay):
    original = replay.confirm(row("Ah Kd Qs Jc Th"))
    for i in range(7):
        partial = row("Ah Kd Qs") if i % 2 else row("Qs Jc Th")
        assert signature(replay(2 + i * 0.6, partial)) == signature(original)
    assert replay(6.25, row("Ah Kd Qs")).board == []


def test_superset_cancels_clearing_even_before_addition_is_confirmed(replay):
    original = replay.confirm()
    for t in [2, 2.6, 3.2, 3.8, 4.4, 5, 5.6]:
        replay(t, [])
    assert signature(replay(6, row("Ah Kd Qs Jc"))) == signature(original)
    for t in [6.6, 7.2, 7.8, 8.4, 9, 9.6, 10.2]:
        assert signature(replay(t, [])) == signature(original)
    assert replay(10.85, []).board == []


def test_unstable_corrections_eventually_clear_unsupported_board(replay):
    original = replay.confirm()
    for i in range(7):
        candidate = row("Ah Kd Jc") if i % 2 else row("Ah Kd Th")
        assert signature(replay(2 + i * 0.6, candidate)) == signature(original)
    assert replay(6.25, row("Ah Kd Jc")).board == []


def test_interrupted_confirmation_starts_fresh(replay):
    replay(0, row())
    replay(0.6, row())
    replay.processor.interrupt()
    assert replay(10, row()).board == []
    assert replay(10.6, row()).board == []
    assert len(replay(11.25, row()).board) == 3


def test_interruption_resets_clearing_but_retains_confirmed_snapshot(replay):
    original = replay.confirm()
    for t in [2, 2.6, 3.2, 3.8, 4.4, 5]:
        replay(t, [])
    replay.processor.interrupt()
    assert signature(replay(10, [])) == signature(original)


def test_detector_errors_do_not_publish_or_change_confirmed_board(replay, monkeypatch):
    original = replay.confirm()
    with monkeypatch.context() as m:

        def fail(_):
            raise RuntimeError("inference failed")

        m.setattr(replay.detector, "process_frame", fail)
        for t in [2, 4, 6, 8, 12]:
            assert replay(t, []) is None
    assert signature(replay(13, row())) == signature(original)


def test_error_interrupts_candidate(replay, monkeypatch):
    replay(0, row())
    replay(0.6, row())
    with monkeypatch.context() as m:
        m.setattr(replay.detector, "process_frame", lambda _: None)
        assert replay(1, []) is None
    assert replay(2, row()).board == []
    assert replay(2.6, row()).board == []
    assert len(replay(3.25, row()).board) == 3


def test_wall_clock_changes_do_not_affect_confirmation(replay, monkeypatch):
    replay(0, row())
    monkeypatch.setattr("floptician.board_processor.time.time", lambda: 10**12)
    assert replay(0.6, row()).board == []
    monkeypatch.setattr("floptician.board_processor.time.time", lambda: -(10**12))
    assert len(replay(1.25, row()).board) == 3


def test_returned_cards_cannot_mutate_confirmed_snapshot(replay):
    original = replay.confirm()
    expected = signature(original)
    original.board[0].x = 99
    original.board.clear()
    assert signature(replay(2, [])) == expected


def test_default_capture_cadence(replay):
    for i in range(3):
        assert replay(i / 1.8, row()).board == []
    assert len(replay(3 / 1.8, row()).board) == 3
    for i in range(8):
        assert len(replay(3 + i / 1.8, []).board) == 3
    assert replay(3 + 8 / 1.8, []).board == []
