from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from floptician.models import BoardConfiguration, CardDetection, CommunityCard, LayoutDescriptor


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


@runtime_checkable
class LayoutMatcher(Protocol):
    descriptor: LayoutDescriptor

    def configuration(self) -> BoardConfiguration: ...

    def matches(self, rows: list[list[CardDetection]], all_detections: list[CardDetection]) -> tuple[bool, dict]: ...

    def assign_coordinates(self, rows: list[list[CardDetection]], context: dict) -> list[CommunityCard]: ...

    def validate(self, community_cards: list[CommunityCard]) -> bool: ...
