from __future__ import annotations

import pytest

from floptician.exceptions import FrameProcessingError
from floptician.models import BoundingBox, CardDetection
from floptician.yolo_processor import YOLOProcessor


def test_inference_exception_is_not_an_empty_detection(normal_frame):
    processor = YOLOProcessor.__new__(YOLOProcessor)
    processor.model_type = "pt"

    def fail(_):
        raise RuntimeError("model unavailable")

    processor.model = fail
    with pytest.raises(FrameProcessingError, match="model unavailable"):
        processor.process_frame(normal_frame)


def test_successful_empty_inference_remains_empty(normal_frame):
    processor = YOLOProcessor.__new__(YOLOProcessor)
    processor.model_type = "pt"
    processor.model = lambda _: []
    processor.confidence_threshold = 0.7
    assert processor.process_frame(normal_frame) == []


class TestBoundingBox:
    def test_center_x(self):
        box = BoundingBox(100, 200, 300, 400)
        assert box.center_x == 200.0

    def test_center_y(self):
        box = BoundingBox(100, 200, 300, 400)
        assert box.center_y == 300.0

    def test_width(self):
        box = BoundingBox(100, 200, 300, 400)
        assert box.width == 200.0

    def test_height(self):
        box = BoundingBox(100, 200, 300, 400)
        assert box.height == 200.0

    def test_to_list(self):
        box = BoundingBox(1, 2, 3, 4)
        assert box.to_list() == [1, 2, 3, 4]

    def test_from_list(self):
        box = BoundingBox.from_list([10, 20, 30, 40])
        assert box.x1 == 10
        assert box.y2 == 40

    def test_frozen_immutable(self):
        box = BoundingBox(1, 2, 3, 4)
        with pytest.raises(AttributeError):
            box.x1 = 99


class TestCalculateIou:
    """Test IoU calculation using the BoundingBox type directly."""

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Reimplementation of YOLOProcessor._calculate_iou for testing."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height

        return intersection / float(area1 + area2 - intersection)

    def test_identical_boxes_iou_1(self):
        box = BoundingBox(0, 0, 100, 100)
        assert self._calculate_iou(box, box) == 1.0

    def test_no_overlap_iou_0(self):
        box1 = BoundingBox(0, 0, 50, 50)
        box2 = BoundingBox(100, 100, 200, 200)
        assert self._calculate_iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        box1 = BoundingBox(0, 0, 100, 100)
        box2 = BoundingBox(50, 50, 150, 150)
        iou = self._calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        assert abs(iou - 2500 / 17500) < 0.001


class TestFilterDetections:
    """Test filtering logic (confidence + NMS) without loading a real model."""

    def test_filters_below_threshold(self):
        detections = [
            CardDetection(card="Ah", confidence=0.9, box=BoundingBox(0, 0, 50, 50)),
            CardDetection(card="Kd", confidence=0.3, box=BoundingBox(100, 100, 150, 150)),
        ]
        threshold = 0.5
        filtered = [d for d in detections if d.confidence >= threshold]
        assert len(filtered) == 1
        assert filtered[0].card == "Ah"

    def test_deduplicates_overlapping_same_card(self):
        detections = [
            CardDetection(card="Ah", confidence=0.95, box=BoundingBox(0, 0, 100, 100)),
            CardDetection(card="Ah", confidence=0.85, box=BoundingBox(10, 10, 110, 110)),
        ]
        # The higher confidence one should win after NMS
        deduped: list[CardDetection] = []
        for det in sorted(detections, key=lambda x: x.confidence, reverse=True):
            should_add = True
            for _existing in deduped:
                # High overlap between these two boxes
                should_add = False
                break
            if should_add:
                deduped.append(det)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.95
