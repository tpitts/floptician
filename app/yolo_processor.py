import logging
import torch
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
        # Check for CUDA device and set it
        self.device = self._select_device()
        self.model = YOLO(model_path, verbose=False).to(self.device)
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

    def _select_device(self) -> str:
        """
        Select the best available device: CUDA if available, otherwise MPS on Apple Silicon, 
        and fallback to CPU.
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif self._detect_apple_silicon() and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def _detect_apple_silicon(self) -> bool:
        """
        Detect if the system is running on Apple Silicon (M1, M1 Pro, M1 Max, M2, etc.).
        """
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def process_frame(self, frame) -> List[Dict[str, Any]]:
        try:
            # Log the dimensions of the 
            target_size = (640, 640)
            frame = self.letterbox_image(frame, target_size)
            frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            frame_height, frame_width = frame.shape[:2]
            logger.debug(f"Processing frame with dimensions: {frame_width}x{frame_height}")

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
    
    def letterbox_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Resize the image to the target size while maintaining the aspect ratio
        and adding padding to fit the target size.
        """
        # Get current size
        height, width = image.shape[:2]

        # Calculate the scale factor and new size
        scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Calculate padding
        delta_w = target_size[0] - new_width
        delta_h = target_size[1] - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Pad the image
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded_image