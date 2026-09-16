"""Synchronous OpenCV camera capture; no subprocesses or application frame queue."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from floptician.camera_devices import CameraDevice, discover_cameras, resolve_camera
from floptician.exceptions import CameraError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureInfo:
    backend: str
    width: int
    height: int
    reported_fps: float


class CameraManager:
    """Owns one capture handle; discovery and OpenCV construction are injectable boundaries.

    Re-resolves device identity before opening. Failed reads close the handle and
    fail explicitly, rather than reconnecting to a potentially different camera.
    All methods must be called on the same thread (the application processing loop).
    """

    def __init__(
        self, swap_channels: bool = False, *, discover: Callable | None = None, capture_factory: Callable | None = None
    ):
        self._discover = discover or discover_cameras
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._capture = None
        self.swap_channels = swap_channels
        self.info: CaptureInfo | None = None
        self.device: CameraDevice | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release_camera()

    def get_available_cameras(self) -> list[CameraDevice]:
        return self._discover()

    def open_camera(self, selected: CameraDevice, *, camera_width=1280, camera_height=720, camera_fps=30) -> None:
        self.release_camera()
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in (camera_width, camera_height)):
            raise CameraError("Camera dimensions must be positive integers")
        if not math.isfinite(camera_fps) or camera_fps <= 0:
            raise CameraError("Camera FPS must be positive and finite")
        self.device = resolve_camera(selected, self._discover())
        try:
            self._capture = self._capture_factory(self.device.index, self.device.backend)
            if not self._capture.isOpened():
                raise CameraError(f"Cannot open '{self.device.name}'. Check camera permissions and other applications.")
            for prop, value in (
                (cv2.CAP_PROP_FRAME_WIDTH, camera_width),
                (cv2.CAP_PROP_FRAME_HEIGHT, camera_height),
                (cv2.CAP_PROP_FPS, camera_fps),
            ):
                if not self._capture.set(prop, value):
                    logger.warning("Camera could not apply property %s=%s; checking delivered frames", prop, value)
            # Verify startup, then discard this frame so the next read gets a fresh sample.
            frame = self._read_frame()
            h, w = frame.shape[:2]
            rate = self._capture.get(cv2.CAP_PROP_FPS)
            rate = rate if math.isfinite(rate) else 0.0
            self.info = CaptureInfo(self._capture.getBackendName(), w, h, rate)
            logger.info(
                "Camera opened: %s | %s | %sx%s, reported %.2f fps (native frames)",
                self.device.name,
                self.info.backend,
                w,
                h,
                rate,
            )
            if (w, h) != (camera_width, camera_height) or abs(rate - camera_fps) > 0.5:
                logger.warning(
                    "Camera negotiated a different mode from requested %sx%s at %s fps",
                    camera_width,
                    camera_height,
                    camera_fps,
                )
        except Exception as exc:
            self.release_camera()
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f"Cannot initialize '{selected.name}': {exc}") from exc

    def _read_frame(self) -> np.ndarray:
        ok, frame = self._capture.read()
        if (
            not ok
            or frame is None
            or frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
        ):
            raise CameraError("Camera stopped delivering BGR frames. Check the connection and restart capture.")
        return frame

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is None:
            return False, None
        try:
            # Preserve native detail; the detector owns the single resize/letterbox.
            frame = self._read_frame()
            if self.swap_channels:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame
        except Exception as exc:
            self.release_camera()
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f"Camera read failed: {exc}") from exc

    def release_camera(self) -> None:
        capture, self._capture = self._capture, None
        self.info = None
        if capture is not None:
            capture.release()
