from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from floptician import camera_devices
from floptician.camera_devices import CameraDevice
from floptician.camera_manager import CameraManager
from floptician.exceptions import CameraError

DEVICE = CameraDevice(2, "Card camera", "persistent-id", cv2.CAP_AVFOUNDATION)


class Capture:
    def __init__(self, frames=None, opened=True):
        self.frames = iter(frames if frames is not None else [np.full((720, 1280, 3), 80, np.uint8)] * 3)
        self.opened = opened
        self.releases = 0
        self.settings = {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def get(self, prop):
        return self.settings[prop]

    def getBackendName(self):
        return "TEST"

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        self.releases += 1


@pytest.mark.parametrize("system,backend", [("Darwin", cv2.CAP_AVFOUNDATION), ("Windows", cv2.CAP_DSHOW)])
def test_discovery_matches_backend_indices(monkeypatch, system, backend):
    monkeypatch.setattr(camera_devices.platform, "system", lambda: system)

    def enumerate_devices(api):
        assert api == backend
        return [SimpleNamespace(index=7, name="C920", path="stable", backend=api)]

    monkeypatch.setattr(camera_devices, "enumerate_cameras", enumerate_devices)
    assert camera_devices.discover_cameras() == [CameraDevice(7, "C920", "stable", backend)]


def test_open_resolves_identity_after_indices_change():
    capture = Capture()
    moved = replace(DEVICE, index=9)
    calls = []

    def factory(index, backend):
        calls.append((index, backend))
        return capture

    with CameraManager(discover=lambda: [moved], capture_factory=factory) as camera:
        camera.open_camera(DEVICE)
        assert calls == [(9, DEVICE.backend)]
        assert camera.info.width == 1280
        ok, frame = camera.get_frame()
        assert ok and frame.shape == (720, 1280, 3)
        assert capture.settings[cv2.CAP_PROP_FPS] == 30
    assert capture.releases == 1


def test_missing_identity_does_not_open_replacement_at_same_index():
    def unexpected_open(*_):
        pytest.fail("Must not open the replacement camera")

    camera = CameraManager(discover=lambda: [replace(DEVICE, device_id="other")], capture_factory=unexpected_open)
    with pytest.raises(CameraError, match="no longer available"):
        camera.open_camera(DEVICE)


@pytest.mark.parametrize("capture", [Capture(opened=False), Capture(frames=[]), Capture(frames=[np.zeros((0, 0, 3))])])
def test_failed_start_releases_capture(capture):
    camera = CameraManager(discover=lambda: [DEVICE], capture_factory=lambda *_: capture)
    with pytest.raises(CameraError):
        camera.open_camera(DEVICE)
    camera.release_camera()
    assert capture.releases == 1
    assert camera.get_frame() == (False, None)


def test_disconnect_closes_handle_and_never_replays_old_frame():
    capture = Capture(frames=[np.full((10, 10, 3), 50, np.uint8)])
    camera = CameraManager(discover=lambda: [DEVICE], capture_factory=lambda *_: capture)
    camera.open_camera(DEVICE)
    with pytest.raises(CameraError, match="stopped delivering"):
        camera.get_frame()
    assert capture.releases == 1
    assert camera.get_frame() == (False, None)


def test_native_frame_is_returned_without_resize_or_copy():
    frame = np.random.default_rng(42).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    capture = Capture(frames=[frame, frame])
    with CameraManager(discover=lambda: [DEVICE], capture_factory=lambda *_: capture) as camera:
        camera.open_camera(DEVICE)
        assert camera.get_frame()[1] is frame


def test_channel_swap_is_explicit():
    frames = [np.full((2, 2, 3), (10, 20, 30), np.uint8)] * 2
    capture = Capture(frames=frames)
    with CameraManager(True, discover=lambda: [DEVICE], capture_factory=lambda *_: capture) as camera:
        camera.open_camera(DEVICE)
        assert np.all(camera.get_frame()[1] == (30, 20, 10))
