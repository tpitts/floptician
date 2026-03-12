from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from floptician.models import CardDetection


@runtime_checkable
class CameraProtocol(Protocol):
    def get_frame(self) -> tuple[bool, np.ndarray | None]: ...

    def release_camera(self) -> None: ...


@runtime_checkable
class OBSProtocol(Protocol):
    def get_version(self): ...

    def get_webcams(self) -> list[str]: ...

    def capture_frame(self, webcam_source: str) -> bytes | None: ...

    def setup_overlay(self) -> None: ...

    def disconnect(self) -> None: ...


@runtime_checkable
class DetectorProtocol(Protocol):
    def process_frame(self, frame: np.ndarray) -> list[CardDetection]: ...


@runtime_checkable
class MessageBroadcaster(Protocol):
    def send_message(self, message: dict, client=None) -> None: ...
