import time
from typing import Dict, List, Any, Tuple
import logging
from collections import deque
from enum import Enum
from app.yolo_processor import YOLOProcessor

logger = logging.getLogger(__name__)

class BoardState(Enum):
    NOT_SHOWING = "Not Showing"
    SHOWING = "Showing"

class TransitionState(Enum):
    NONE = "None"
    APPEARING = "Appearing"
    DISAPPEARING = "Disappearing"

class BoardProcessor:
    MAX_HISTORY_SIZE = 1000  # Adjust if memory usage becomes an issue
    MIN_CARDS_IN_ROW = 3  # Constant value because board is always at least 3 cards

    def __init__(self, config: Dict[str, Any]):
        self.yolo_processor = YOLOProcessor(config['yolo']['model'], 
                                            config['yolo']['confidence_threshold'],
                                            config['yolo']['overlap_threshold'])
        self.config = config['board_processor']
        self.frame_id = 0
        self.board_state = BoardState.NOT_SHOWING
        self.transition_state = TransitionState.NONE
        self.detection_history = deque(maxlen=self.MAX_HISTORY_SIZE)
        self.last_state_change = time.time()
        self.stable_board = []
        self.transition_start_time = None

    def process_frame(self, frame) -> Dict[str, Any]:
        """
        Process a single video frame to detect community cards and determine board state.
        Returns a dictionary with detection results and debug information.
        """
        self.frame_id += 1
        start_time = time.time()
        logger.info('processing frame!!!!!')
        try:
            filtered_detections = self.yolo_processor.process_frame(frame) or []  # Ensure it's always a list
            detected_board = self._detect_community_cards(filtered_detections)
            
            new_state, new_transition_state, detected_board, displayed_board = self._update_board_state(detected_board)
            
            processing_time = time.time() - start_time

            # Remove bounding boxes from detected board
            for card_info in detected_board:
                if 'box' in card_info:
                    del card_info['box']
            
            # Remove bounding boxes from displayed board
            for card_info in displayed_board:
                if 'box' in card_info:
                    del card_info['box']

            # Return format must remain consistent for backwards compatibility
            return {
                "timestamp": time.time(),
                "state": new_state,  # Return the BoardState enum object
                "board": displayed_board,
                "debug_info": {
                    "detections": [
                        {"card": d['card'], "confidence": d['confidence']}
                        for d in filtered_detections
                    ],
                    "processing_time": processing_time,
                    "frame_id": self.frame_id,
                    "current_state": new_state,  # Return the BoardState enum object
                    "transition_state": new_transition_state,  # Include the TransitionState
                    "detected_board": detected_board
                }
            }
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {
                "timestamp": time.time(),
                "state": BoardState.NOT_SHOWING,  # Return the BoardState enum object
                "board": [],
                "debug_info": {
                    "current_state": BoardState.NOT_SHOWING,
                    "transition_state": TransitionState.NONE,
                    "error": str(e)
                }
            }
        
    def _detect_community_cards(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify community cards from detections by grouping them into rows based on vertical alignment.
        """
        if not detections:
            return []

        def calculate_center(box):
            x1, y1, x2, y2 = box
            return [(x1 + x2) / 2, (y1 + y2) / 2]

        centers = [(d, calculate_center(d['box'])) for d in detections]
        if not centers:
            return []

        centers = sorted(centers, key=lambda c: c[1][0])  # Sort by x-coordinate

        rows = []
        current_row = [centers[0]]

        for (detection, center) in centers[1:]:
            prev_center = current_row[-1][1]
            box_height = detection['box'][3] - detection['box'][1]
            threshold = box_height * self.config['vertical_alignment_threshold']

            if abs(center[1] - prev_center[1]) < threshold:
                current_row.append((detection, center))
            else:
                rows.append(current_row)
                current_row = [(detection, center)]

        rows.append(current_row)

        community_cards = []
        for row in rows:
            if len(row) >= self.MIN_CARDS_IN_ROW:
                community_cards.extend([detection for detection, center in row])

        return community_cards

    def _update_board_state(self, detected_board: List[Dict]) -> Tuple[BoardState, TransitionState, List[Dict], List[Dict]]:
        """
        Update the state machine based on current detections and history.
        Returns the current board state, transition state, detected board, and displayed board.
        """
        self.detection_history.append(detected_board)
        logger.debug(f"Updating board state. Current: {self.board_state}, Transition: {self.transition_state}")

        if self.board_state == BoardState.NOT_SHOWING:
            if self.transition_state == TransitionState.NONE:
                if detected_board:
                    logger.debug("Transitioning from NOT_SHOWING/NONE to NOT_SHOWING/APPEARING")
                    self.transition_state = TransitionState.APPEARING
                    self.last_state_change = time.time()
                else:
                    logger.debug("Remaining in NOT_SHOWING/NONE state")
            elif self.transition_state == TransitionState.DISAPPEARING:
                logger.debug("Unexpected state: NOT_SHOWING/DISAPPEARING. Resetting to NOT_SHOWING/NONE")
                self.transition_state = TransitionState.NONE
                self.last_state_change = time.time()
            elif self.transition_state == TransitionState.APPEARING:
                if not detected_board or self._board_has_missing_cards(self.stable_board, detected_board):
                    logger.debug("Reverting to NOT_SHOWING/NONE due to missing cards during APPEARING")
                    self.board_state, self.transition_state = self._revert_to_previous_state(detected_board)
                    self.last_state_change = time.time()
                elif self._meets_appearance_thresholds(detected_board):
                    logger.debug("Appearance thresholds met, transitioning to SHOWING/NONE")
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
                    logger.debug("Removal thresholds met, transitioning to NOT_SHOWING/NONE")
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
                        logger.debug("Appearance thresholds met, transitioning to SHOWING/NONE")
                        self.board_state = BoardState.SHOWING
                        self.transition_state = TransitionState.NONE
                        self.stable_board = detected_board
                        self.last_state_change = time.time()
                else: 
                    logger.debug("Remaining in SHOWING/APPEARING state")

        logger.info(f"Updated state: {self.board_state}/{self.transition_state}, Detected board: {[card['card'] for card in detected_board]}")
        
        return self.board_state, self.transition_state, detected_board, self.stable_board

    def _transition_to(self, new_board_state: BoardState, new_transition_state: TransitionState, new_board: List[Dict]) -> Tuple[BoardState, TransitionState, List[Dict], List[Dict]]:
        """
        Handle state transitions, update internal state, and log the transition.
        """
        self.board_state = new_board_state
        self.transition_state = new_transition_state
        self.last_state_change = time.time()
        
        if new_board_state == BoardState.SHOWING and new_transition_state == TransitionState.NONE:
            self.stable_board = new_board
        
        displayed_board = self.stable_board if new_transition_state == TransitionState.DISAPPEARING else new_board
        
        logger.info(f"Transitioning to {self._get_combined_state()} state. New board: {[card['card'] for card in new_board]}")
        return new_board_state, new_transition_state, new_board, displayed_board

    def _revert_to_previous_state(self, detected_board: List[Dict]) -> Tuple[BoardState, TransitionState]:
        """
        Revert to the previous stable state when a transition fails.
        """
        if self.board_state == BoardState.NOT_SHOWING:
            return BoardState.NOT_SHOWING, TransitionState.NONE
        else:
            return BoardState.SHOWING, TransitionState.NONE

    def _get_combined_state(self) -> str:
        """
        Get a string representation of the combined board and transition state.
        """
        if self.transition_state == TransitionState.NONE:
            return self.board_state.value
        return f"{self.board_state.value}_{self.transition_state.value}"

    def _boards_match(self, board1: List[Dict], board2: List[Dict]) -> bool:
        """
        Check if two boards match exactly.
        """
        return set(card['card'] for card in board1) == set(card['card'] for card in board2)

    def _board_has_missing_cards(self, board1: List[Dict], board2: List[Dict]) -> bool:
        """
        Check if any cards from board1 are missing in board2.
        """
        return any(card['card'] not in [c['card'] for c in board2] for card in board1)

    def _board_has_new_cards(self, board1: List[Dict], board2: List[Dict]) -> bool:
        """
        Check if board1 has any new cards compared to board2, without any removals.
        """
        cards1 = set(card['card'] for card in board1)
        cards2 = set(card['card'] for card in board2)
        return cards1.issuperset(cards2) and cards1 != cards2

    def _meets_appearance_thresholds(self, board: List[Dict]) -> bool:
        result = self._is_new_state_established(self.config['min_frames_to_show'], self.config['min_ms_to_show'], board)
        logger.debug(f"Appearance thresholds met: {result}")
        return result


    def _meets_removal_thresholds(self) -> bool:
        return self._is_consistent_removal(self.config['min_frames_to_remove'], self.config['min_ms_to_remove'])
    
    def _is_consistent_removal(self, min_frames: int, min_ms: int) -> bool:
        if len(self.detection_history) < min_frames:
            logger.debug(f"Not enough frames in history: {len(self.detection_history)} < {min_frames}")
            return False

        stable_card_set = set(card['card'] for card in self.stable_board)

        for i, board in enumerate(list(self.detection_history)[-min_frames:], 1):
            current_card_set = set(card['card'] for card in board)
            if current_card_set == stable_card_set:
                logger.debug(f"Frame {i}: Detected board matches stable board")
                return False

        time_consistent = (time.time() - self.last_state_change) >= (min_ms / 1000)
        logger.debug(f"Time consistent: {time_consistent} (Elapsed time: {time.time() - self.last_state_change}s, Required: {min_ms / 1000}s)")
        
        return time_consistent

    def _is_new_state_established(self, min_frames: int, min_ms: int, target_board: List[Dict]) -> bool:
        """
        Check if a new state has been consistently detected for a specified number of frames and duration.
        This helps prevent rapid state changes due to momentary detection errors.
        
        We only want to evaluate the card values, not other attributes like confidence and bounding box.
        """
        if len(self.detection_history) < min_frames:
            logger.debug(f"Not enough frames in history: {len(self.detection_history)} < {min_frames}")
            return False

        # Create sets of card identifiers (ignoring bounding boxes and confidence)
        target_card_set = set(card['card'] for card in target_board)

        for i, board in enumerate(list(self.detection_history)[-min_frames:], 1):
            current_card_set = set(card['card'] for card in board)
            if current_card_set != target_card_set:
                logger.debug(f"Frame {i}: Detected board {current_card_set} does not match target board {target_card_set}")
                return False

        time_consistent = (time.time() - self.last_state_change) >= (min_ms / 1000)
        logger.debug(f"Time consistent: {time_consistent} (Elapsed time: {time.time() - self.last_state_change}s, Required: {min_ms / 1000}s)")
        
        return time_consistent

    def _remove_bounding_boxes(self, board: List[Dict]) -> List[Dict]:
        """
        Remove bounding box information from the board representation.
        """
        return [{k: v for k, v in card.items() if k != 'box'} for card in board]

    def _format_board(self, cards: List[Dict]) -> List[Dict]:
        """
        Format the detected cards into a standardized board representation,
        always ordered from left to right based on bounding box information.
        """
        if not cards:
            return []
        sorted_cards = sorted(cards, key=lambda c: c['box'][0])
        return [
            {"card": card['card'], "x": i, "y": 1}
            for i, card in enumerate(sorted_cards)
        ]