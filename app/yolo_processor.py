import logging
import os
import cv2
import numpy as np
import platform
from ultralytics import YOLO
from typing import Dict, List, Any

# Set YOLOv8 to quiet mode
os.environ['YOLO_VERBOSE'] = 'False'
logger = logging.getLogger(__name__)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

class YOLOProcessor:
    def __init__(self, model_path: str, confidence_threshold: float, overlap_threshold: float):
        self.model = YOLO(model_path, verbose=False)
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold
        self.is_apple_silicon = self._detect_apple_silicon()

    def _detect_apple_silicon(self) -> bool:
        """
        Detect if the system is running on Apple Silicon (M1, M1 Pro, M1 Max, M2, etc.).
        """
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def process_frame(self, frame) -> List[Dict[str, Any]]:
        try:
            # Log the dimensions of the frame
            frame_height, frame_width = frame.shape[:2]
            logger.debug(f"Processing frame with dimensions: {frame_width}x{frame_height}")

            # Process the frame with the YOLO model using the appropriate device
            if self.is_apple_silicon:
                results = self.model(frame, device="mps")
            else:
                results = self.model(frame)  # YOLO will select the best available device

            detections = self._extract_detections(results)
            filtered_detections = self._filter_detections(detections)
            return filtered_detections
        except Exception as e:
            logger.error(f"Error processing frame with YOLO: {str(e)}")
            return []

    def _extract_detections(self, results) -> List[Dict[str, Any]]:
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
        filtered = [d for d in detections if d['confidence'] >= self.confidence_threshold]
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
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection / float(area1 + area2 - intersection)
        return iou
