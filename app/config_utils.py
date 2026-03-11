import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
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


def find_config_path(config_path: Optional[str] = None) -> Path:
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

    raise FileNotFoundError(
        f"No config file found. Expected {DEFAULT_CONFIG_PATH} or {EXAMPLE_CONFIG_PATH}."
    )


def _normalize_config_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    config_dir = config_path.parent

    for key in ("html_file", "output_dir"):
        if isinstance(config.get(key), str):
            config[key] = str(resolve_path(config[key], config_dir))

    yolo_config = config.get("yolo")
    if isinstance(yolo_config, dict):
        if "mcoreml_model" in yolo_config and "coreml_model" not in yolo_config:
            yolo_config["coreml_model"] = yolo_config["mcoreml_model"]

        for key in ("model", "coreml_model", "mcoreml_model"):
            if isinstance(yolo_config.get(key), str):
                yolo_config[key] = str(resolve_path(yolo_config[key], config_dir))

    config["_config_path"] = str(config_path)
    return config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    resolved_path = find_config_path(config_path)

    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing config file {resolved_path}: {exc}") from exc

    if not config:
        raise ValueError(f"Config file is empty: {resolved_path}")

    return _normalize_config_paths(config, resolved_path)
