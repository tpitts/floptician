"""Custom exception hierarchy for Floptician."""

from __future__ import annotations


class FlopticiannError(Exception):
    """Base exception for all Floptician errors."""


class ConfigurationError(FlopticiannError):
    """Error in configuration loading or validation."""


class CameraError(FlopticiannError):
    """Error related to camera access or frame capture."""


class OBSConnectionError(FlopticiannError):
    """Error connecting to or communicating with OBS."""


class ModelLoadError(FlopticiannError):
    """Error loading the YOLO model."""


class ServerStartupError(FlopticiannError):
    """Error starting HTTP or WebSocket servers."""


class FrameProcessingError(FlopticiannError):
    """Error during frame processing pipeline."""
