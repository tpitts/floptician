from __future__ import annotations

import logging
import os
import platform

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from floptician.exceptions import FrameProcessingError, ModelLoadError
from floptician.models import BoundingBox, CardDetection, YOLOConfig

# Set YOLOv8 to quiet mode
os.environ["YOLO_VERBOSE"] = "False"
logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class YOLOProcessor:
    def __init__(self, config: YOLOConfig):
        self.device = self._select_device()
        self.confidence_threshold = config.confidence_threshold
        self.overlap_threshold = config.overlap_threshold

        if config.model.endswith(".pt"):
            self.model_type = "pt"
            self.model = YOLO(config.model, verbose=False).to(self.device)
        elif config.model.endswith(".mlpackage"):
            self.model_type = "mlpackage"
            self.model = self._load_coreml_model(config.model)
        else:
            raise ModelLoadError("Unsupported model format. Please provide a .pt or .mlpackage file.")

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif self._detect_apple_silicon() and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _detect_apple_silicon(self) -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def _load_coreml_model(self, model_path: str):
        try:
            import coremltools as ct
        except ImportError as e:
            raise ModelLoadError("coremltools is required to use .mlpackage models.") from e

        try:
            model = ct.models.MLModel(model_path)
            logger.info(f"Successfully loaded Core ML model from {model_path}")
            logger.info(f"Core ML model input names: {model.input_description}")
            logger.info(f"Core ML model output names: {model.output_description}")
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load Core ML model from {model_path}: {e}") from e

    def process_frame(self, frame) -> list[CardDetection]:
        try:
            target_size = (640, 640)
            frame = self.letterbox_image(frame, target_size)
            frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            frame_height, frame_width = frame.shape[:2]
            logger.debug(f"Processing frame with dimensions: {frame_width}x{frame_height}")

            if self.model_type == "pt":
                results = self.model(frame)
                detections = self._extract_detections(results)
            elif self.model_type == "mlpackage":
                input_image = Image.fromarray(frame)
                input_data = {"image": input_image}
                result = self.model.predict(input_data)
                detections = self._extract_detections_coreml(result)
            else:
                raise ModelLoadError("Unsupported model type.")

            return self._filter_detections(detections)
        except Exception as e:
            raise FrameProcessingError(f"Error processing frame with YOLO: {e}") from e

    def _extract_detections(self, results) -> list[CardDetection]:
        detections = []
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf, strict=False):
                class_index = int(cls.item())
                class_name = result.names[class_index]
                detections.append(
                    CardDetection(
                        card=class_name,
                        confidence=round(float(conf), 3),
                        box=BoundingBox.from_list(box.tolist()),
                    )
                )
        return detections

    def _extract_detections_coreml(self, result) -> list[CardDetection]:
        detections = []
        output_key = "var_1140"
        output_data = result[output_key]

        logger.debug(f"Output type: {type(output_data)}")
        logger.debug(f"Output shape: {output_data.shape}")
        logger.debug(f"Output data: {output_data}")

        if isinstance(output_data, np.ndarray):
            flattened_output = output_data.reshape(-1, 8400)

            for detection in flattened_output:
                class_id = int(detection[0])
                confidence = float(detection[1])
                bbox = detection[2:6].tolist()

                if confidence >= self.confidence_threshold:
                    detections.append(
                        CardDetection(
                            card=str(class_id),
                            confidence=round(confidence, 3),
                            box=BoundingBox.from_list(bbox),
                        )
                    )
        else:
            raise FrameProcessingError("Unsupported output data format from Core ML model.")

        return detections

    def _filter_detections(self, detections: list[CardDetection]) -> list[CardDetection]:
        filtered = [d for d in detections if d.confidence >= self.confidence_threshold]
        deduped: list[CardDetection] = []
        for detection in sorted(filtered, key=lambda x: x.confidence, reverse=True):
            should_add = True
            for existing in deduped:
                if self._calculate_iou(detection.box, existing.box) > self.overlap_threshold:
                    should_add = False
                    break
            if should_add:
                deduped.append(detection)
        return deduped

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height

        return intersection / float(area1 + area2 - intersection)

    def letterbox_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        delta_w = target_size[0] - new_width
        delta_h = target_size[1] - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        return cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
