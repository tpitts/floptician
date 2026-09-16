"""Device discovery, using the same index ordering as each OpenCV backend."""

from __future__ import annotations

import platform
from dataclasses import dataclass

import cv2
from cv2_enumerate_cameras import enumerate_cameras

from floptician.exceptions import CameraError


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    device_id: str
    backend: int


def discover_cameras() -> list[CameraDevice]:
    backend = {
        "Darwin": cv2.CAP_AVFOUNDATION,
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
    }.get(platform.system())
    if backend is None:
        raise CameraError(f"Camera capture is unsupported on {platform.system()}")
    try:
        devices = enumerate_cameras(backend)
        return [CameraDevice(d.index, d.name, str(d.path), d.backend) for d in devices if d.path]
    except Exception as exc:
        raise CameraError(f"Cannot enumerate cameras: {exc}") from exc


def resolve_camera(selected: CameraDevice, devices: list[CameraDevice]) -> CameraDevice:
    matches = [d for d in devices if (d.backend, d.device_id) == (selected.backend, selected.device_id)]
    if len(matches) != 1:
        raise CameraError(f"Camera '{selected.name}' is no longer available. Reconnect it and select it again.")
    return matches[0]
