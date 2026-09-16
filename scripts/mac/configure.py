"""First-run Mac configuration; existing settings are never silently replaced."""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from floptician.config_utils import load_config, validate_config

ROOT = Path(__file__).resolve().parents[2]


def read_mapping(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML settings mapping in {path}")
    return data


def imported_settings(old: dict) -> dict:
    # Only import settings whose meaning survives the 2024 migration. Model,
    # overlay paths and device indexes must not point into the old installation.
    settings = {"yolo": {"model": "auto"}, "capture": {"mode": "direct_webcam"}}
    sections = {
        "obs": ("host", "port", "password"),
        "capture": ("fps", "camera_width", "camera_height", "camera_fps"),
        "yolo": ("confidence_threshold", "overlap_threshold"),
        "board_processor": (
            "vertical_alignment_threshold",
            "min_frames_to_show",
            "min_ms_to_show",
            "min_frames_to_remove",
            "min_ms_to_remove",
        ),
    }
    for section, keys in sections.items():
        source = old.get(section, {})
        if not isinstance(source, dict):
            raise ValueError(f"Old config section {section!r} must be a mapping")
        settings.setdefault(section, {}).update({key: source[key] for key in keys if key in source})
    for key in ("http_port", "websocket_port"):
        if key in old:
            settings[key] = old[key]
    return settings


def backup_config(path: Path, root: Path) -> Path:
    backups = root / "output" / "setup-backups"
    backups.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="config-", dir=backups))
    target = directory / "config.yaml"
    shutil.copy2(path, target)
    target.chmod(0o600)
    return target


def create_config(root: Path, ask=input, secret=getpass.getpass) -> None:
    target = root / "config.yaml"
    if target.exists():
        print("Keeping existing config.yaml unchanged.")
        return
    source = ask("Old config.yaml path to import (drag it here, or Enter for new settings): ").strip().strip("'\"")
    # Finder may escape spaces when a file is dragged into Terminal.
    source = source.replace("\\ ", " ")
    settings = imported_settings(read_mapping(Path(source).expanduser())) if source else imported_settings({})
    if source:
        backup = backup_config(Path(source).expanduser(), root)
        print(f"Old settings backed up to {backup}")
        print("Imported OBS credentials, detection rate/thresholds and ports.")
        print("Using native camera capture and the current recommended model; old paths/indexes are not imported.")
    if not settings.get("obs", {}).get("password"):
        print("OBS: Tools > WebSocket Server Settings > Enable WebSocket server > Show Connect Info.")
        password = secret("OBS WebSocket password (hidden; Enter if authentication is disabled): ")
        settings.setdefault("obs", {})["password"] = password
    # Exclusive creation protects an existing file, even if another writer raced us.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        yaml.safe_dump(settings, file, sort_keys=False)
    print("Created private config.yaml. Future updates will preserve it.")


def model_include(root: Path) -> str:
    model = Path(load_config(str(root / "config.yaml")).yolo.model)
    try:
        relative = model.relative_to(root).as_posix()
    except ValueError:
        return ""  # Explicit external models belong to the user.
    if any(char in relative for char in "*,?[]"):
        raise ValueError("Model path contains Git LFS wildcard characters; use a simple model filename.")
    return relative + "/**" if model.suffix == ".mlpackage" else relative


def smoke_test(root: Path) -> None:
    import cv2

    from floptician.yolo_processor import YOLOProcessor

    config = load_config(str(root / "config.yaml"))
    validate_config(config)
    print("Checking model load and a real inference (no camera access required)...")
    frame = cv2.imread(str(root / "tests/screenshots/single_row/turn001.png"))
    if frame is None:
        raise ValueError("The setup test image is missing. Retry Update.")
    detections = YOLOProcessor(config.yolo).process_frame(frame)
    if not detections:
        raise ValueError("The model did not detect cards in the setup test image.")
    print(f"Model check passed: {len(detections)} card detections.")
    import obsws_python as obs

    client = None
    try:
        client = obs.ReqClient(host=config.obs.host, port=config.obs.port, password=config.obs.password, timeout=3)
        version = client.get_version()
        print(f"OBS connection passed: OBS {version.obs_version}, WebSocket {version.obs_web_socket_version}.")
    except Exception:
        # OBS may legitimately be closed during installation. Never print credentials.
        print("OBS is not ready: open OBS, enable its WebSocket server, and check the password.")
        print("To change the saved password, open Setup Floptician.command and choose option 2.")
    finally:
        if client is not None:
            client.disconnect()
    print("Live camera/OBS coexistence still needs a first run with your camera and OBS open.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--model-include", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--obs-password", action="store_true")
    args = parser.parse_args()
    try:
        if args.model_include:
            print(model_include(ROOT))
        elif args.smoke_test:
            smoke_test(ROOT)
        elif args.obs_password:
            path = ROOT / "config.yaml"
            settings = read_mapping(path)
            password = getpass.getpass("New OBS WebSocket password: ")
            backup = backup_config(path, ROOT)
            settings.setdefault("obs", {})["password"] = password
            fd, temporary = tempfile.mkstemp(dir=ROOT, prefix=".config-", suffix=".yaml")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    yaml.safe_dump(settings, file, sort_keys=False)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            print(f"Password updated. Previous settings: {backup}")
        else:
            create_config(ROOT)
    except (ValueError, OSError, EOFError) as exc:
        parser.exit(1, f"Setup error: {exc}\n")


if __name__ == "__main__":
    main()
