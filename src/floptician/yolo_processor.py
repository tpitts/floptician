from __future__ import annotations

import ast
import logging
import os
import platform

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from floptician.exceptions import FrameProcessingError, ModelLoadError
from floptician.models import DEFAULT_YOLO_IMAGE_SIZE, BoundingBox, CardDetection, YOLOConfig

# Set YOLOv8 to quiet mode
os.environ["YOLO_VERBOSE"] = "False"
logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class YOLOProcessor:
    def __init__(self, config: YOLOConfig):
        self.device = self._select_device()
        self.inference_size = config.image_size
        self.coreml_compute_unit = config.coreml_compute_unit
        self.confidence_threshold = config.confidence_threshold
        self.overlap_threshold = config.overlap_threshold

        if config.model.endswith(".pt"):
            self.model_type = "pt"
            self.model = YOLO(config.model, verbose=False).to(self.device)
            self.input_size = (640, 640)
            self.class_names = dict(self.model.names)
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
            compute_units = {
                "all": ct.ComputeUnit.ALL,
                "cpu-only": ct.ComputeUnit.CPU_ONLY,
                "cpu-and-gpu": ct.ComputeUnit.CPU_AND_GPU,
                "cpu-and-ne": ct.ComputeUnit.CPU_AND_NE,
            }
            requested_compute_unit = compute_units[self.coreml_compute_unit]
            try:
                model = ct.models.MLModel(model_path, compute_units=requested_compute_unit)
            except Exception:
                if requested_compute_unit == ct.ComputeUnit.ALL:
                    raise
                logger.warning(
                    "Core ML compute unit %s is unavailable; falling back to all compute units",
                    self.coreml_compute_unit,
                    exc_info=True,
                )
                model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
            spec = model.get_spec()
            image_inputs = [item for item in spec.description.input if item.type.WhichOneof("Type") == "imageType"]
            if len(image_inputs) != 1:
                raise ValueError(f"expected exactly one image input, found {len(image_inputs)}")

            output_names = {item.name for item in spec.description.output}
            required_outputs = {"confidence", "coordinates"}
            if not required_outputs.issubset(output_names):
                raise ValueError(
                    "Core ML model must be exported with embedded NMS and provide "
                    f"{sorted(required_outputs)}; found {sorted(output_names)}"
                )

            image_input = image_inputs[0]
            self.coreml_image_input = image_input.name
            self.coreml_input_names = {item.name for item in spec.description.input}
            self.input_size = (image_input.type.imageType.width, image_input.type.imageType.height)
            if self.input_size != (self.inference_size, self.inference_size):
                raise ValueError(
                    f"Core ML input is {self.input_size[0]}x{self.input_size[1]}, "
                    f"but config requests {self.inference_size}x{self.inference_size}"
                )
            self.class_names = self._parse_coreml_class_names(model.user_defined_metadata.get("names"))
            logger.info(
                "Loaded Core ML model %s (input=%s, size=%sx%s, classes=%s)",
                model_path,
                self.coreml_image_input,
                self.input_size[0],
                self.input_size[1],
                len(self.class_names),
            )
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load Core ML model from {model_path}: {e}") from e

    def process_frame(self, frame) -> list[CardDetection]:
        try:
            target_size = getattr(self, "input_size", (640, 640))
            frame = self.letterbox_image(frame, target_size)
            frame_height, frame_width = frame.shape[:2]
            logger.debug(f"Processing frame with dimensions: {frame_width}x{frame_height}")

            if self.model_type == "pt":
                results = self.model(frame, imgsz=getattr(self, "inference_size", DEFAULT_YOLO_IMAGE_SIZE))
                detections = self._extract_detections(results)
            elif self.model_type == "mlpackage":
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_image = Image.fromarray(rgb_frame)
                input_data = {self.coreml_image_input: input_image}
                if "confidenceThreshold" in self.coreml_input_names:
                    input_data["confidenceThreshold"] = self.confidence_threshold
                if "iouThreshold" in self.coreml_input_names:
                    input_data["iouThreshold"] = self.overlap_threshold
                result = self.model.predict(input_data)
                detections = self._extract_detections_coreml(result, frame_width, frame_height)
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

    @staticmethod
    def _parse_coreml_class_names(raw_names: str | None) -> dict[int, str]:
        if not raw_names:
            raise ValueError("Core ML model metadata is missing class names")
        parsed = ast.literal_eval(raw_names)
        if isinstance(parsed, list):
            return dict(enumerate(str(name) for name in parsed))
        if isinstance(parsed, dict):
            return {int(index): str(name) for index, name in parsed.items()}
        raise ValueError("Core ML class names metadata must be a list or dictionary")

    def _extract_detections_coreml(self, result: dict, frame_width: int, frame_height: int) -> list[CardDetection]:
        confidences = result.get("confidence")
        coordinates = result.get("coordinates")
        if not isinstance(confidences, np.ndarray) or not isinstance(coordinates, np.ndarray):
            raise FrameProcessingError("Core ML output must contain confidence and coordinates arrays")
        if confidences.ndim != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise FrameProcessingError(
                f"Unexpected Core ML output shapes: confidence={confidences.shape}, coordinates={coordinates.shape}"
            )
        if confidences.shape[0] != coordinates.shape[0]:
            raise FrameProcessingError("Core ML confidence and coordinate row counts do not match")

        detections = []
        for scores, (center_x, center_y, width, height) in zip(confidences, coordinates, strict=True):
            class_index = int(np.argmax(scores))
            confidence = float(scores[class_index])
            if confidence < self.confidence_threshold:
                continue
            card = self.class_names.get(class_index)
            if card is None:
                raise FrameProcessingError(f"Core ML returned unknown class index {class_index}")
            box = [
                (center_x - width / 2) * frame_width,
                (center_y - height / 2) * frame_height,
                (center_x + width / 2) * frame_width,
                (center_y + height / 2) * frame_height,
            ]
            detections.append(CardDetection(card=card, confidence=round(confidence, 3), box=BoundingBox.from_list(box)))
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
