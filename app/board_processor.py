import time
from typing import Dict, List, Any, Tuple
import logging
from ultralytics import YOLO
import cv2
import numpy as np

# Set up logger for debugging and error reporting
logger = logging.getLogger(__name__)

class BoardProcessor:
    """
    Processes video frames to detect and stabilize community cards using a YOLO model.

    This class handles detecting cards, filtering false positives, identifying community cards,
    and stabilizing the detection over multiple frames. The resulting community card data
    is then sent to a webpage used as an overlay in OBS.
    """

    def __init__(self, model_path: str):
        """
        Initialize the BoardProcessor with the given YOLO model path.
        Set up necessary attributes for detection and stabilization.
        """
        self.model = YOLO(model_path)
        self.stable_board = []  # List of cards in the currently stabilized board
        self.stable_frame_count = 0  # Number of consecutive frames with the same board
        self.frames_to_stabilize = 2  # Number of consecutive frames required for stabilization
        self.frame_id = 0  # Counter for processed frames
        self.confidence_threshold = 0.70  # Minimum confidence score for detections
        self.overlap_threshold = 0.80  # IoU threshold for removing overlapping detections

    def process_frame(self, frame) -> Dict[str, Any]:
        """
        Process a single video frame to detect community cards and determine board state.
        Returns a dictionary with detection results and debug information.
        """
        self.frame_id += 1
        start_time = time.time()

        try:
            # Run YOLO model for object detection
            results = self.model(frame) 

            # Extract and filter detections
            detections = self._extract_detections(results)
            filtered_detections = self._filter_detections(detections)

            # Identify community cards
            community_cards = self._detect_community_cards(filtered_detections)

            # Determine board state and stabilize
            state, board = self._get_board_state(community_cards)

            processing_time = time.time() - start_time

            return {
                "timestamp": time.time(),
                "state": state,
                "board": board,
                "debug_info": {
                    "detections": [
                        {"card": d['card'], "confidence": d['confidence']}
                        for d in filtered_detections
                    ],
                    "processing_time": processing_time,
                    "frame_id": self.frame_id
                }
            }
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {
                "timestamp": time.time(),
                "state": "error",
                "error": str(e)
            }

    def _extract_detections(self, results) -> List[Dict[str, Any]]:
        """
        Extract detections from YOLO model results.
        Returns a list of dictionaries with card names, confidence scores, and bounding boxes.
        """
        detections = []
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                class_index = int(cls.item())
                class_name = result.names[class_index]
                detections.append({
                    'card': class_name,
                    'confidence': round(float(conf), 3),
                    'box': box.tolist()
                })
        return detections

    def _filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections based on confidence and deduplicate overlapping detections.
        Returns a list of filtered and deduplicated detections.
        """
        # Filter by confidence threshold
        filtered = [d for d in detections if d['confidence'] >= self.confidence_threshold]

        # Remove overlapping detections (Non-Maximum Suppression)
        deduped = []
        for detection in sorted(filtered, key=lambda x: x['confidence'], reverse=True):
            should_add = True
            for existing in deduped:
                if self._calculate_iou(detection['box'], existing['box']) > self.overlap_threshold:
                    should_add = False
                    break
            if should_add:
                deduped.append(detection)
        return deduped

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        Returns the IoU value.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection / float(area1 + area2 - intersection)
        return iou

    def _detect_community_cards(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify community cards from detections by grouping them into rows based on vertical alignment.
        Returns a list of detected community cards.
        """
        def calculate_center(box):
            x1, y1, x2, y2 = box
            return [(x1 + x2) / 2, (y1 + y2) / 2]

        centers = [(d, calculate_center(d['box'])) for d in detections]
        if not centers:
            return []

        centers = sorted(centers, key=lambda c: c[1][0])  # Sort by x-coordinate

        rows = []
        current_row = [centers[0]]

        # Group detections into rows based on vertical alignment
        for (detection, center) in centers[1:]:
            prev_center = current_row[-1][1]
            box_height = detection['box'][3] - detection['box'][1]
            threshold = box_height * 0.25  # Vertical alignment threshold

            if abs(center[1] - prev_center[1]) < threshold:
                current_row.append((detection, center))
            else:
                rows.append(current_row)
                current_row = [(detection, center)]

        rows.append(current_row)

        # Identify rows with at least 3 cards as community cards
        community_cards = []
        for row in rows:
            if len(row) >= 3:  # Minimum number of cards to consider a row
                community_cards.extend([detection for detection, center in row])

        return community_cards

    def _get_board_state(self, community_cards: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Determine the state of the board based on the detected community cards.
        Returns the board state and the list of stable community cards if stabilized.
        """
        if not community_cards:
            self.stable_frame_count = 0
            return "not_detected", []

        # Check if the detected cards are the same as the previous stable board
        if self._is_same_board(community_cards, self.stable_board):
            self.stable_frame_count += 1
        else:
            self.stable_frame_count = 0
            self.stable_board = community_cards

        # If the same board is detected for enough consecutive frames, mark it as stable
        if self.stable_frame_count >= self.frames_to_stabilize:
            return "stable", [
                {"card": card['card'], "x": i, "y": 1}
                for i, card in enumerate(community_cards)
            ]
        
        return "detected", []

    def _is_same_board(self, board1: List[Dict], board2: List[Dict]) -> bool:
        """
        Check if two boards contain the same set of cards.
        Returns True if they are the same, False otherwise.
        """
        return set(card['card'] for card in board1) == set(card['card'] for card in board2)