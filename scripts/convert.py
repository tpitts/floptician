"""Export a Floptician YOLO detector to a validated Core ML package."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import coremltools as ct
from _common import models_path
from ultralytics import YOLO

from floptician.models import DEFAULT_YOLO_IMAGE_SIZE, DEFAULT_YOLO_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=models_path(Path(DEFAULT_YOLO_MODEL).name),
        help="PyTorch .pt model to export (defaults to the current Floptician model)",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_YOLO_IMAGE_SIZE, help="Square Core ML input size")
    return parser.parse_args()


def validate_export(path: Path, expected_size: int) -> None:
    model = ct.models.MLModel(str(path))
    spec = model.get_spec()
    image_inputs = [item for item in spec.description.input if item.type.WhichOneof("Type") == "imageType"]
    outputs = {item.name for item in spec.description.output}
    if len(image_inputs) != 1:
        raise RuntimeError(f"Expected one image input, found {len(image_inputs)}")
    image_type = image_inputs[0].type.imageType
    if (image_type.width, image_type.height) != (expected_size, expected_size):
        raise RuntimeError(f"Unexpected input size: {image_type.width}x{image_type.height}")
    if not {"confidence", "coordinates"}.issubset(outputs):
        raise RuntimeError(f"Export does not contain embedded-NMS outputs: {sorted(outputs)}")
    names = ast.literal_eval(model.user_defined_metadata.get("names", ""))
    if not isinstance(names, (dict, list)) or not names:
        raise RuntimeError("Export is missing class-name metadata")


def main() -> None:
    args = parse_args()
    source = args.model.expanduser().resolve()
    if source.suffix != ".pt" or not source.is_file():
        raise SystemExit(f"Model must be an existing .pt file: {source}")

    exported = Path(
        YOLO(str(source)).export(format="coreml", imgsz=args.imgsz, nms=True, device="cpu", half=False, int8=False)
    ).resolve()
    validate_export(exported, args.imgsz)
    print(f"Validated Core ML export: {exported}")


if __name__ == "__main__":
    main()
