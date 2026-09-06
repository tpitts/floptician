from __future__ import annotations

from types import SimpleNamespace

from floptician import camera_manager


class FakeCapture:
    def __init__(self):
        self.set_calls: list[tuple[int, int]] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: int) -> bool:
        self.set_calls.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


class TestOpenCamera:
    def test_open_camera_uses_cap_prop_fps(self, monkeypatch):
        fake_capture = FakeCapture()
        fake_cv2 = SimpleNamespace(
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FPS=5,
            VideoCapture=lambda *_args: fake_capture,
        )

        monkeypatch.setattr(camera_manager, "cv2", fake_cv2)
        monkeypatch.setattr(camera_manager, "is_macos", lambda: False)
        monkeypatch.setattr(camera_manager, "is_windows", lambda: False)
        monkeypatch.setattr(camera_manager, "windows_dshow", False)
        monkeypatch.setattr(camera_manager.CameraManager, "refresh_camera_list", lambda self: [])

        manager = camera_manager.CameraManager()

        assert manager.open_camera(8, width=1280, height=720, fps=30) is True
        assert fake_capture.set_calls == [(3, 1280), (4, 720), (5, 30)]

    def test_open_camera_skips_fps_when_property_missing(self, monkeypatch, caplog):
        fake_capture = FakeCapture()
        fake_cv2 = SimpleNamespace(
            CAP_PROP_FRAME_WIDTH=10,
            CAP_PROP_FRAME_HEIGHT=11,
            VideoCapture=lambda *_args: fake_capture,
        )

        monkeypatch.setattr(camera_manager, "cv2", fake_cv2)
        monkeypatch.setattr(camera_manager, "is_macos", lambda: False)
        monkeypatch.setattr(camera_manager, "is_windows", lambda: False)
        monkeypatch.setattr(camera_manager, "windows_dshow", False)
        monkeypatch.setattr(camera_manager.CameraManager, "refresh_camera_list", lambda self: [])

        manager = camera_manager.CameraManager()

        assert manager.open_camera(2, width=640, height=480, fps=24) is True
        assert fake_capture.set_calls == [(10, 640), (11, 480)]
        assert "skipping FPS configuration" in caplog.text
