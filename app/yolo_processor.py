import logging
import torch
import os
import cv2
import numpy as np
import platform
from ultralytics import YOLO
import coremltools as ct
from PIL import Image
from typing import Dict, List, Any

# Set YOLOv8 to quiet mode
os.environ['YOLO_VERBOSE'] = 'False'
logger = logging.getLogger(__name__)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

class YOLOProcessor:
    def __init__(self, model_path: str, confidence_threshold: float, overlap_threshold: float):
        # Check for CUDA device and set it
        self.device = self._select_device()
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

        # Determine the model type and load the appropriate model
        if model_path.endswith('.pt'):
            self.model_type = 'pt'
            self.model = YOLO(model_path, verbose=False).to(self.device)
        elif model_path.endswith('.mlpackage'):
            self.model_type = 'mlpackage'
            self.model = self._load_coreml_model(model_path)
        else:
            raise ValueError("Unsupported model format. Please provide a .pt or .mlpackage file.")

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

    def _load_coreml_model(self, model_path: str):
        """
        Load the Core ML model.
        """
        try:
            model = ct.models.MLModel(model_path)
            logger.info(f"Successfully loaded Core ML model from {model_path}")

            # Log input and output names for debugging purposes
            logger.info(f"Core ML model input names: {model.input_description}")
            logger.info(f"Core ML model output names: {model.output_description}")

            return model
        except Exception as e:
            logger.error(f"Failed to load Core ML model from {model_path}: {e}")
            raise

    def process_frame(self, frame) -> List[Dict[str, Any]]:
        try:
            # Log the dimensions of the frame
            target_size = (640, 640)
            frame = self.letterbox_image(frame, target_size)
            frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            frame_height, frame_width = frame.shape[:2]
            logger.debug(f"Processing frame with dimensions: {frame_width}x{frame_height}")

            if self.model_type == 'pt':
                # Use PyTorch model for inference
                results = self.model(frame)
                detections = self._extract_detections(results)
            elif self.model_type == 'mlpackage':
                # Use Core ML model for inference
                input_image = Image.fromarray(frame)
                input_data = {'image': input_image}  # Use the correct input key
                result = self.model.predict(input_data)
                detections = self._extract_detections_coreml(result)
            else:
                raise ValueError("Unsupported model type.")

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

    def _extract_detections_coreml(self, result) -> List[Dict[str, Any]]:
        """
        Extract detections from Core ML model output.
        """
        detections = []
        # Use the correct output name from your Core ML model
        output_key = 'var_1140'
        output_data = result[output_key]

        # Log the type, shape, and content of the output data for debugging
        logger.debug(f"Output type: {type(output_data)}")
        logger.debug(f"Output shape: {output_data.shape}")
        logger.debug(f"Output data: {output_data}")

        # Handle the output data based on its type
        if isinstance(output_data, np.ndarray):
            # Flatten the output to ensure correct indexing
            flattened_output = output_data.reshape(-1, 8400)  # Adjust based on your output structure
            
            for detection in flattened_output:
                # Assuming the detection array contains the following:
                # [class, confidence, x1, y1, x2, y2, ...]
                class_id = int(detection[0])  # Access class ID
                confidence = float(detection[1])  # Access confidence score
                bbox = detection[2:6].tolist()  # Access bounding box coordinates [x1, y1, x2, y2]
                
                # Only add detections above a confidence threshold
                if confidence >= self.confidence_threshold:
                    detections.append({
                        'card': class_id,
                        'confidence': round(confidence, 3),
                        'box': bbox
                    })
        else:
            logger.error("Unsupported output data format from Core ML model.")
    
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
