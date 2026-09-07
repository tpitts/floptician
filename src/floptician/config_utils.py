from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml

from floptician.exceptions import ConfigurationError
from floptician.models import (
    DEFAULT_YOLO_IMAGE_SIZE,
    DEFAULT_YOLO_MODEL,
    AppConfig,
    BoardProcessorConfig,
    CaptureConfig,
    CaptureMode,
    OBSConfig,
    YOLOConfig,
)

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
        swap_channels=cap_raw.get("swap_channels", False),
    )

    # YOLO
    yolo_raw = raw.get("yolo", {})
    model_path = yolo_raw.get("model", DEFAULT_YOLO_MODEL)
    model_path = str(resolve_path(model_path, config_dir))
    yolo = YOLOConfig(
        model=model_path,
        image_size=yolo_raw.get("image_size", DEFAULT_YOLO_IMAGE_SIZE),
        coreml_compute_unit=yolo_raw.get("coreml_compute_unit", "cpu-and-ne"),
        confidence_threshold=yolo_raw.get("confidence_threshold", 0.70),
        overlap_threshold=yolo_raw.get("overlap_threshold", 0.80),
    )

    # Board processor
    bp_raw = raw.get("board_processor", {})
    board_processor = BoardProcessorConfig(
        vertical_alignment_threshold=bp_raw.get("vertical_alignment_threshold", 0.20),
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


def _check_number(name: str, value: Any, minimum: float, maximum: float = math.inf, *, integer=False) -> None:
    expected_types = (int,) if integer else (int, float)
    if type(value) not in expected_types or not math.isfinite(value) or not minimum <= value <= maximum:
        kind = "integer" if integer else "number"
        raise ConfigurationError(f"{name} must be a finite {kind} in [{minimum}, {maximum}], got {value!r}")


def validate_config(config: AppConfig) -> None:
    _check_number("YOLO confidence_threshold", config.yolo.confidence_threshold, 0, 1)
    _check_number("YOLO overlap_threshold", config.yolo.overlap_threshold, 0, 1)
    _check_number("YOLO image_size", config.yolo.image_size, 32, integer=True)
    if config.yolo.image_size % 32:
        raise ConfigurationError(f"YOLO image_size must be divisible by 32, got {config.yolo.image_size}")
    valid_compute_units = {"all", "cpu-only", "cpu-and-gpu", "cpu-and-ne"}
    if config.yolo.coreml_compute_unit not in valid_compute_units:
        raise ConfigurationError(
            f"YOLO coreml_compute_unit must be one of {sorted(valid_compute_units)}, "
            f"got {config.yolo.coreml_compute_unit!r}"
        )
    if not (0.0 < config.yolo.confidence_threshold <= 1.0):
        raise ConfigurationError(f"YOLO confidence_threshold must be in (0, 1], got {config.yolo.confidence_threshold}")
    if not (0.0 < config.yolo.overlap_threshold <= 1.0):
        raise ConfigurationError(f"YOLO overlap_threshold must be in (0, 1], got {config.yolo.overlap_threshold}")
    _check_number("Capture FPS", config.capture.fps, 0)
    if config.capture.fps <= 0:
        raise ConfigurationError(f"Capture FPS must be positive, got {config.capture.fps}")
    for name in ("width", "height"):
        _check_number(f"Capture {name}", getattr(config.capture, name), 1, integer=True)
    for name, port in (
        ("http_port", config.http_port),
        ("websocket_port", config.websocket_port),
        ("OBS port", config.obs.port),
    ):
        _check_number(name, port, 1, 65535, integer=True)
    if config.http_port == config.websocket_port:
        raise ConfigurationError("http_port and websocket_port must be different")
    bp = config.board_processor
    _check_number("vertical_alignment_threshold", bp.vertical_alignment_threshold, 0)
    if bp.vertical_alignment_threshold == 0:
        raise ConfigurationError("vertical_alignment_threshold must be positive")
    for name in ("min_frames_to_show", "min_frames_to_remove"):
        _check_number(name, getattr(bp, name), 1, integer=True)
    for name in ("min_ms_to_show", "min_ms_to_remove"):
        _check_number(name, getattr(bp, name), 0, integer=True)
    model_path = Path(config.yolo.model)
    if not model_path.exists():
        raise ConfigurationError(f"YOLO model file not found: {model_path}")
    if not Path(config.html_file).is_file():
        raise ConfigurationError(f"Overlay HTML file not found: {config.html_file}")


def load_config(config_path: str | None = None) -> AppConfig:
    resolved_path = find_config_path(config_path)

    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Error parsing config file {resolved_path}: {exc}") from exc

    if raw is None or raw == {}:
        raise ConfigurationError(f"Config file is empty: {resolved_path}")
    if not isinstance(raw, dict):
        raise ConfigurationError("Config must contain a YAML mapping of settings")
    for section in ("obs", "capture", "yolo", "board_processor"):
        if section in raw and not isinstance(raw[section], dict):
            raise ConfigurationError(f"Config section {section!r} must be a mapping of settings")
    try:
        return _build_app_config(raw, resolved_path)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid config value: {exc}") from exc
