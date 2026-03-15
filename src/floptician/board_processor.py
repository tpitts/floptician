from __future__ import annotations

import logging
import time
from collections import deque

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
from floptician.yolo_processor import YOLOProcessor

logger = logging.getLogger(__name__)


class BoardProcessor:
    MAX_HISTORY_SIZE = 1000

    def __init__(self, config: AppConfig, detector: DetectorProtocol | None = None):
        self.yolo_processor: DetectorProtocol = detector or YOLOProcessor(config.yolo)
        self.config = config.board_processor
        self.community_card_detector = CommunityCardDetector(self.config)
        self.frame_id = 0
        self.board_state = BoardState.NOT_SHOWING
        self.transition_state = TransitionState.NONE
        self.detection_history: deque[list[CommunityCard]] = deque(maxlen=self.MAX_HISTORY_SIZE)
        self.last_state_change = time.time()
        self.stable_board: list[CommunityCard] = []
        self.current_configuration = BoardConfiguration.NO_BOARD
        self.transition_start_time: float | None = None

    def process_frame(self, frame) -> BoardResult:
        self.frame_id += 1
        start_time = time.time()
        try:
            filtered_detections = self.yolo_processor.process_frame(frame) or []
            detected_board, configuration = self.community_card_detector.detect_community_cards(filtered_detections)
            self.current_configuration = configuration
            sorted_board = sorted(detected_board, key=lambda card: (card.y, card.x))
            card_values = [card.card for card in sorted_board]
            logger.debug(f"Detected board: {card_values}")
            new_state, new_transition_state, detected_board, displayed_board = self._update_board_state(detected_board)

            # Update x and y coordinates of the stable board
            updated_displayed_board = self._update_positions(displayed_board, detected_board)

            processing_time = time.time() - start_time

            return BoardResult(
                timestamp=time.time(),
                state=new_state,
                board=updated_displayed_board,
                configuration=self.current_configuration,
                debug_info={
                    "detections": [{"card": d.card, "confidence": d.confidence} for d in filtered_detections],
                    "processing_time": processing_time,
                    "frame_id": self.frame_id,
                    "current_state": new_state.value,
                    "transition_state": new_transition_state.value,
                    "detected_board": [
                        {"card": c.card, "x": c.x, "y": c.y, "confidence": c.confidence} for c in detected_board
                    ],
                },
            )
        except Exception as e:
            logger.error(f"Error processing frame: {e!s}")
            return BoardResult(
                timestamp=time.time(),
                state=BoardState.NOT_SHOWING,
                board=[],
                debug_info={
                    "current_state": BoardState.NOT_SHOWING.value,
                    "transition_state": TransitionState.NONE.value,
                    "error": str(e),
                },
            )

    def _update_positions(
        self, stable_board: list[CommunityCard], detected_board: list[CommunityCard]
    ) -> list[CommunityCard]:
        updated_board: list[CommunityCard] = []
        for stable_card in stable_board:
            for detected_card in detected_board:
                if stable_card.card == detected_card.card:
                    updated_board.append(
                        CommunityCard(
                            card=stable_card.card,
                            x=detected_card.x,
                            y=detected_card.y,
                            confidence=stable_card.confidence,
                        )
                    )
                    break
            else:
                updated_board.append(stable_card)
        return updated_board

    def _update_board_state(
        self, detected_board: list[CommunityCard]
    ) -> tuple[BoardState, TransitionState, list[CommunityCard], list[CommunityCard]]:
        self.detection_history.append(detected_board)

        if self.board_state == BoardState.NOT_SHOWING:
            if self.transition_state == TransitionState.NONE:
                if detected_board:
                    logger.debug("Transitioning from NOT_SHOWING/NONE to NOT_SHOWING/APPEARING")
                    self.transition_state = TransitionState.APPEARING
                    self.last_state_change = time.time()
                else:
                    logger.debug("Remaining in NOT_SHOWING/NONE state")
            elif self.transition_state == TransitionState.DISAPPEARING:
                logger.error("Unexpected state: NOT_SHOWING/DISAPPEARING. Resetting to NOT_SHOWING/NONE")
                self.transition_state = TransitionState.NONE
                self.last_state_change = time.time()
            elif self.transition_state == TransitionState.APPEARING:
                if not detected_board or self._board_has_missing_cards(self.stable_board, detected_board):
                    logger.debug("Reverting to NOT_SHOWING/NONE due to missing cards during APPEARING")
                    self.board_state, self.transition_state = self._revert_to_previous_state()
                    self.last_state_change = time.time()
                elif self._meets_appearance_thresholds(detected_board):
                    logger.info("Appearance thresholds met, showing cards")
                    self.board_state = BoardState.SHOWING
                    self.transition_state = TransitionState.NONE
                    self.stable_board = detected_board
                    self.last_state_change = time.time()
                else:
                    logger.debug("Remaining in NOT_SHOWING/APPEARING state")

        elif self.board_state == BoardState.SHOWING:
            if self.transition_state == TransitionState.NONE:
                if not detected_board or self._board_has_missing_cards(self.stable_board, detected_board):
                    logger.debug("Transitioning from SHOWING/NONE to SHOWING/DISAPPEARING")
                    self.transition_state = TransitionState.DISAPPEARING
                    self.last_state_change = time.time()
                elif self._board_has_new_cards(detected_board, self.stable_board):
                    logger.debug("Transitioning from SHOWING/NONE to SHOWING/APPEARING")
                    self.transition_state = TransitionState.APPEARING
                    self.last_state_change = time.time()
                else:
                    logger.debug("Remaining in SHOWING/NONE state")
            elif self.transition_state == TransitionState.DISAPPEARING:
                if self._boards_match(detected_board, self.stable_board):
                    logger.debug("Transitioning from SHOWING/DISAPPEARING to SHOWING/NONE")
                    self.transition_state = TransitionState.NONE
                    self.last_state_change = time.time()
                elif self._meets_removal_thresholds():
                    logger.info("Removal thresholds met, clearing the cards")
                    self.board_state = BoardState.NOT_SHOWING
                    self.transition_state = TransitionState.NONE
                    self.last_state_change = time.time()
                    self.stable_board = []
                else:
                    logger.debug("Remaining in SHOWING/DISAPPEARING state")
            elif self.transition_state == TransitionState.APPEARING:
                if not detected_board or self._board_has_missing_cards(self.stable_board, detected_board):
                    logger.debug("Transitioning from SHOWING/APPEARING to SHOWING/DISAPPEARING")
                    self.transition_state = TransitionState.DISAPPEARING
                    self.last_state_change = time.time()
                elif self._boards_match(detected_board, self.stable_board):
                    logger.debug("Transitioning from SHOWING/APPEARING to SHOWING/NONE")
                    self.transition_state = TransitionState.NONE
                    self.last_state_change = time.time()
                elif self._meets_appearance_thresholds(detected_board):
                    logger.info("Appearance thresholds met, updating with new cards")
                    self.board_state = BoardState.SHOWING
                    self.transition_state = TransitionState.NONE
                    self.stable_board = detected_board
                    self.last_state_change = time.time()
                else:
                    logger.debug("Remaining in SHOWING/APPEARING state")

        return self.board_state, self.transition_state, detected_board, self.stable_board

    def _revert_to_previous_state(self) -> tuple[BoardState, TransitionState]:
        if self.board_state == BoardState.NOT_SHOWING:
            return BoardState.NOT_SHOWING, TransitionState.NONE
        else:
            return BoardState.SHOWING, TransitionState.NONE

    def _get_combined_state(self) -> str:
        if self.transition_state == TransitionState.NONE:
            return self.board_state.value
        return f"{self.board_state.value}_{self.transition_state.value}"

    def _boards_match(self, board1: list[CommunityCard], board2: list[CommunityCard]) -> bool:
        return {card.card for card in board1} == {card.card for card in board2}

    def _board_has_missing_cards(self, board1: list[CommunityCard], board2: list[CommunityCard]) -> bool:
        board2_cards = {c.card for c in board2}
        return any(card.card not in board2_cards for card in board1)

    def _board_has_new_cards(self, board1: list[CommunityCard], board2: list[CommunityCard]) -> bool:
        cards1 = {card.card for card in board1}
        cards2 = {card.card for card in board2}
        return cards1.issuperset(cards2) and cards1 != cards2

    def _meets_appearance_thresholds(self, board: list[CommunityCard]) -> bool:
        result = self._is_new_state_established(self.config.min_frames_to_show, self.config.min_ms_to_show, board)
        logger.debug(f"Appearance thresholds met: {result}")
        return result

    def _meets_removal_thresholds(self) -> bool:
        return self._is_consistent_removal(self.config.min_frames_to_remove, self.config.min_ms_to_remove)

    def _is_consistent_removal(self, min_frames: int, min_ms: int) -> bool:
        if len(self.detection_history) < min_frames:
            logger.debug(f"Not enough frames in history: {len(self.detection_history)} < {min_frames}")
            return False

        stable_card_set = {card.card for card in self.stable_board}

        for i, board in enumerate(list(self.detection_history)[-min_frames:], 1):
            current_card_set = {card.card for card in board}
            if current_card_set == stable_card_set:
                logger.debug(f"Frame {i}: Detected board matches stable board")
                return False

        time_consistent = (time.time() - self.last_state_change) >= (min_ms / 1000)
        elapsed = time.time() - self.last_state_change
        logger.debug(f"Time consistent: {time_consistent} (Elapsed: {elapsed}s, Required: {min_ms / 1000}s)")

        return time_consistent

    def _is_new_state_established(self, min_frames: int, min_ms: int, target_board: list[CommunityCard]) -> bool:
        if len(self.detection_history) < min_frames:
            logger.debug(f"Not enough frames in history: {len(self.detection_history)} < {min_frames}")
            return False

        target_card_set = {card.card for card in target_board}

        for i, board in enumerate(list(self.detection_history)[-min_frames:], 1):
            current_card_set = {card.card for card in board}
            if current_card_set != target_card_set:
                logger.debug(f"Frame {i}: Detected board {current_card_set} does not match target {target_card_set}")
                return False

        time_consistent = (time.time() - self.last_state_change) >= (min_ms / 1000)
        elapsed = time.time() - self.last_state_change
        logger.debug(f"Time consistent: {time_consistent} (Elapsed: {elapsed}s, Required: {min_ms / 1000}s)")

        return time_consistent
