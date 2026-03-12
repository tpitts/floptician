from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from floptician.models import AppConfig, BoardProcessorConfig, CaptureConfig, CaptureMode, OBSConfig, YOLOConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "config.example.yaml"


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def _resolve_config_candidate(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def find_config_path(config_path: str | None = None) -> Path:
    if config_path:
        candidate = _resolve_config_candidate(config_path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Config file not found: {candidate}")

    env_path = os.getenv("FLOPTICIAN_CONFIG")
    if env_path:
        candidate = _resolve_config_candidate(env_path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Config file from FLOPTICIAN_CONFIG not found: {candidate}")

    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    if EXAMPLE_CONFIG_PATH.is_file():
        return EXAMPLE_CONFIG_PATH

    raise FileNotFoundError(f"No config file found. Expected {DEFAULT_CONFIG_PATH} or {EXAMPLE_CONFIG_PATH}.")


def _build_app_config(raw: dict[str, Any], config_path: Path) -> AppConfig:
    config_dir = config_path.parent

    # Resolve paths relative to config dir
    html_file = str(resolve_path(raw.get("html_file", "src/floptician/static/overlay.html"), config_dir))
    output_dir = str(resolve_path(raw.get("output_dir", "output"), config_dir))

    # OBS
    obs_raw = raw.get("obs", {})
    obs_password = os.getenv("OBS_PASSWORD", obs_raw.get("password", ""))
    obs = OBSConfig(
        host=obs_raw.get("host", "localhost"),
        port=obs_raw.get("port", 4455),
        password=obs_password,
    )

    # Capture
    cap_raw = raw.get("capture", {})
    capture = CaptureConfig(
        mode=CaptureMode(cap_raw.get("mode", "direct_webcam")),
        width=cap_raw.get("width", 640),
        height=cap_raw.get("height", 360),
        fps=cap_raw.get("fps", 1.8),
    )

    # YOLO
    yolo_raw = raw.get("yolo", {})
    model_path = yolo_raw.get("model", "models/yolov8l-2026-03-10.pt")
    model_path = str(resolve_path(model_path, config_dir))
    yolo = YOLOConfig(
        model=model_path,
        confidence_threshold=yolo_raw.get("confidence_threshold", 0.70),
        overlap_threshold=yolo_raw.get("overlap_threshold", 0.80),
    )

    # Board processor
    bp_raw = raw.get("board_processor", {})
    board_processor = BoardProcessorConfig(
        vertical_alignment_threshold=bp_raw.get("vertical_alignment_threshold", 0.20),
        horizontal_alignment_threshold=bp_raw.get("horizontal_alignment_threshold", 0.10),
        image_height=bp_raw.get("image_height", 1080),
        min_frames_to_show=bp_raw.get("min_frames_to_show", 3),
        min_ms_to_show=bp_raw.get("min_ms_to_show", 1200),
        min_frames_to_remove=bp_raw.get("min_frames_to_remove", 6),
        min_ms_to_remove=bp_raw.get("min_ms_to_remove", 4200),
    )

    return AppConfig(
        debug=raw.get("debug", False),
        host=raw.get("host", "localhost"),
        http_port=raw.get("http_port", 8000),
        websocket_port=raw.get("websocket_port", 9001),
        html_file=html_file,
        output_dir=output_dir,
        config_path=str(config_path),
        obs=obs,
        capture=capture,
        yolo=yolo,
        board_processor=board_processor,
    )


def validate_config(config: AppConfig) -> None:
    if not (0.0 < config.yolo.confidence_threshold <= 1.0):
        raise ValueError(f"YOLO confidence_threshold must be in (0, 1], got {config.yolo.confidence_threshold}")
    if not (0.0 < config.yolo.overlap_threshold <= 1.0):
        raise ValueError(f"YOLO overlap_threshold must be in (0, 1], got {config.yolo.overlap_threshold}")
    model_path = Path(config.yolo.model)
    if not model_path.exists():
        raise ValueError(f"YOLO model file not found: {model_path}")
    if config.capture.fps <= 0:
        raise ValueError(f"Capture FPS must be positive, got {config.capture.fps}")


def load_config(config_path: str | None = None) -> AppConfig:
    resolved_path = find_config_path(config_path)

    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing config file {resolved_path}: {exc}") from exc

    if not raw:
        raise ValueError(f"Config file is empty: {resolved_path}")

    return _build_app_config(raw, resolved_path)
