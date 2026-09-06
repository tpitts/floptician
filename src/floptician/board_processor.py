from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from floptician.community_card_detector import CommunityCardDetector
from floptician.models import (
    AppConfig,
    BoardConfiguration,
    BoardResult,
    BoardState,
    CommunityCard,
    TransitionState,
)
from floptician.protocols import DetectorProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BoardSnapshot:
    configuration: BoardConfiguration = BoardConfiguration.NO_BOARD
    cards: tuple[CommunityCard, ...] = ()

    @classmethod
    def from_cards(cls, cards: list[CommunityCard], configuration: BoardConfiguration) -> _BoardSnapshot:
        if not cards:
            return cls()
        return cls(configuration, tuple(replace(c) for c in sorted(cards, key=lambda c: (c.y, c.x, c.card))))

    @property
    def key(self) -> tuple:
        # Confidence and camera pixel coordinates are not part of board identity.
        return self.configuration, tuple((c.card, c.x, c.y) for c in self.cards)

    @property
    def card_names(self) -> set[str]:
        return {c.card for c in self.cards}

    def copy_cards(self) -> list[CommunityCard]:
        return [replace(c) for c in self.cards]


class BoardProcessor:
    def __init__(
        self,
        config: AppConfig,
        detector: DetectorProtocol | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        if detector is None:
            from floptician.yolo_processor import YOLOProcessor

            detector = YOLOProcessor(config.yolo)
        self.yolo_processor = detector
        self.config = config.board_processor
        self.community_card_detector = CommunityCardDetector(self.config)
        self.frame_id = 0
        self._clock = clock
        self._confirmed = _BoardSnapshot()
        self._candidate: _BoardSnapshot | None = None
        self._candidate_since: float | None = None
        self._candidate_frames = 0
        self._missing_since: float | None = None
        self._missing_frames = 0

    @property
    def board_state(self) -> BoardState:
        return BoardState.SHOWING if self._confirmed.cards else BoardState.NOT_SHOWING

    @property
    def transition_state(self) -> TransitionState:
        if self._candidate is not None:
            return TransitionState.APPEARING
        if self._missing_since is not None:
            return TransitionState.DISAPPEARING
        return TransitionState.NONE

    @property
    def stable_board(self) -> list[CommunityCard]:
        return self._confirmed.copy_cards()

    @property
    def current_configuration(self) -> BoardConfiguration:
        return self._confirmed.configuration

    def process_frame(self, frame) -> BoardResult | None:
        self.frame_id += 1
        start_time = self._clock()
        try:
            detections = self.yolo_processor.process_frame(frame)
            if detections is None:
                raise ValueError("Detector returned no result")
            cards, configuration = self.community_card_detector.detect_community_cards(detections)
            observed = _BoardSnapshot.from_cards(cards, configuration)
            self._observe(observed, self._clock())
        except Exception:
            self.interrupt()
            logger.exception("Error processing frame; retaining the confirmed board")
            # Do not broadcast an error as a successful empty observation. The browser's
            # existing inactivity timeout must still expire during a prolonged outage.
            return None

        return BoardResult(
            timestamp=time.time(),
            state=self.board_state,
            board=self.stable_board,
            configuration=self.current_configuration,
            debug_info={
                "detections": [{"card": d.card, "confidence": d.confidence} for d in detections],
                "processing_time": self._clock() - start_time,
                "frame_id": self.frame_id,
                "current_state": self.board_state.value,
                "transition_state": self.transition_state.value,
                "detected_board": [{"card": c.card, "x": c.x, "y": c.y, "confidence": c.confidence} for c in cards],
            },
        )

    def interrupt(self) -> None:
        """An unusable/missing frame breaks agreement, without changing the displayed board."""
        self._reset_candidate()
        self._missing_since = None
        self._missing_frames = 0

    def _reset_candidate(self) -> None:
        self._candidate = None
        self._candidate_since = None
        self._candidate_frames = 0

    def _observe(self, observed: _BoardSnapshot, now: float) -> None:
        if observed.key == self._confirmed.key:
            self.interrupt()
            return

        confirmed_names = self._confirmed.card_names
        observed_names = observed.card_names
        if confirmed_names - observed_names:
            if self._missing_since is None:
                self._missing_since = now
            self._missing_frames += 1
        else:
            # Seeing all confirmed cards cancels clearing, even with unconfirmed extras.
            self._missing_since = None
            self._missing_frames = 0

        # A strict subset is partial disappearance, not a correction. Preserve the whole
        # board until the slower removal gate is met. Other nonempty candidates, including
        # identity corrections and layout changes, all use the same appearance gate.
        if observed.cards and not observed_names < confirmed_names:
            if self._candidate is None or observed.key != self._candidate.key:
                self._candidate = observed
                self._candidate_since = now
                self._candidate_frames = 1
            else:
                self._candidate_frames += 1

            if (
                self._candidate_frames >= self.config.min_frames_to_show
                and self._candidate_since is not None
                and now - self._candidate_since >= self.config.min_ms_to_show / 1000
            ):
                self._confirmed = observed
                self.interrupt()
                logger.info("Confirmed board: %s", observed.key)
                return
        else:
            self._reset_candidate()

        if (
            self._missing_frames >= self.config.min_frames_to_remove
            and self._missing_since is not None
            and now - self._missing_since >= self.config.min_ms_to_remove / 1000
        ):
            self._confirmed = _BoardSnapshot()
            self.interrupt()
            logger.info("Removal thresholds met, clearing the board")
