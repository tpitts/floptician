#!/usr/bin/env python3
"""One-off script: run YOLO on all screenshots, print detections for manual curation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from floptician.models import YOLOConfig
from floptician.yolo_processor import YOLOProcessor

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "tests" / "screenshots"


def main():
    config = YOLOConfig()
    print(f"Loading model: {config.model}")
    processor = YOLOProcessor(config)

    for png in sorted(SCREENSHOTS_DIR.rglob("*.png")):
        rel = png.relative_to(SCREENSHOTS_DIR)
        print(f"\n{'='*60}")
        print(f"  {rel}")
        print(f"{'='*60}")

        frame = cv2.imread(str(png))
        if frame is None:
            print("  ERROR: could not read image")
            continue

        detections = processor.process_frame(frame)
        detections.sort(key=lambda d: (d.box.center_y, d.box.center_x))

        for d in detections:
            print(
                f"  {d.card:>3s}  conf={d.confidence:.3f}  "
                f"center=({d.box.center_x:.0f}, {d.box.center_y:.0f})  "
                f"box=({d.box.x1:.0f}, {d.box.y1:.0f}, {d.box.x2:.0f}, {d.box.y2:.0f})  "
                f"size={d.box.width:.0f}x{d.box.height:.0f}"
            )

        if not detections:
            print("  (no detections)")


if __name__ == "__main__":
    main()
