"""Compare Core ML and PyTorch against every labeled screenshot."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
from _common import models_path, repo_path

from floptician.community_card_detector import CommunityCardDetector
from floptician.models import DEFAULT_YOLO_MODEL, BoardProcessorConfig, YOLOConfig
from floptician.yolo_processor import YOLOProcessor


def parse_args() -> argparse.Namespace:
    default_pt = models_path(Path(DEFAULT_YOLO_MODEL).name)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch", type=Path, default=default_pt)
    parser.add_argument("--coreml", type=Path, default=default_pt.with_suffix(".mlpackage"))
    parser.add_argument("--screenshots", type=Path, default=repo_path("tests", "screenshots"))
    parser.add_argument(
        "--compute-unit",
        choices=("all", "cpu-only", "cpu-and-gpu", "cpu-and-ne"),
        default="all",
        help="Core ML compute-unit policy",
    )
    return parser.parse_args()


def run(processor: YOLOProcessor, frame, detector: CommunityCardDetector):
    started = time.perf_counter()
    detections = processor.process_frame(frame)
    elapsed = time.perf_counter() - started
    cards, configuration = detector.detect_community_cards(detections)
    board = {(card.card, card.x, card.y) for card in cards}
    return detections, board, configuration.name, elapsed


def main() -> None:
    args = parse_args()
    pt = YOLOProcessor(YOLOConfig(model=str(args.pytorch.resolve())))
    coreml = YOLOProcessor(YOLOConfig(model=str(args.coreml.resolve())))
    if args.compute_unit != "all":
        import coremltools as ct

        compute_units = {
            "cpu-only": ct.ComputeUnit.CPU_ONLY,
            "cpu-and-gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu-and-ne": ct.ComputeUnit.CPU_AND_NE,
        }
        coreml.model = ct.models.MLModel(str(args.coreml.resolve()), compute_units=compute_units[args.compute_unit])
    detector = CommunityCardDetector(BoardProcessorConfig())
    failures: list[str] = []
    pt_times: list[float] = []
    coreml_times: list[float] = []

    cases = sorted(args.screenshots.rglob("*.expected.json"))
    if not cases:
        raise SystemExit(f"No expected screenshot files found under {args.screenshots}")

    for expected_path in cases:
        image_path = expected_path.with_name(expected_path.name.replace(".expected.json", ".png"))
        frame = cv2.imread(str(image_path))
        if frame is None:
            failures.append(f"{image_path}: could not read image")
            continue
        expected_data = json.loads(expected_path.read_text(encoding="utf-8"))
        expected_board = {(card["card"], card["x"], card["y"]) for card in expected_data["cards"]}
        expected_configuration = expected_data["configuration"]

        pt_detections, pt_board, pt_configuration, pt_elapsed = run(pt, frame, detector)
        ml_detections, ml_board, ml_configuration, ml_elapsed = run(coreml, frame, detector)
        pt_times.append(pt_elapsed)
        coreml_times.append(ml_elapsed)

        relative = image_path.relative_to(args.screenshots)
        for backend, board, configuration in (
            ("PyTorch", pt_board, pt_configuration),
            ("Core ML", ml_board, ml_configuration),
        ):
            if board != expected_board or configuration != expected_configuration:
                failures.append(
                    f"{relative} [{backend}]: expected {expected_configuration} {sorted(expected_board)}, "
                    f"got {configuration} {sorted(board)}"
                )
        pt_cards = sorted(detection.card for detection in pt_detections)
        ml_cards = sorted(detection.card for detection in ml_detections)
        if pt_cards != ml_cards:
            failures.append(f"{relative} [raw parity]: PyTorch {pt_cards}, Core ML {ml_cards}")

    print(f"Validated {len(cases)} labeled screenshots")
    print(f"Core ML compute units: {args.compute_unit}")
    print(f"PyTorch median inference: {statistics.median(pt_times) * 1000:.1f} ms")
    print(f"Core ML median inference: {statistics.median(coreml_times) * 1000:.1f} ms")
    if failures:
        print("\nParity failures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Core ML and PyTorch parity passed")


if __name__ == "__main__":
    main()
