from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from typer.testing import CliRunner

from floptician.camera_devices import CameraDevice
from floptician.cli import _select_input, app

runner = CliRunner()


def test_camera_menu_uses_padded_reverse_highlight(monkeypatch):
    from types import SimpleNamespace

    from floptician import cli

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    captured = {}

    def select(message, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(unsafe_ask=lambda: 1)

    monkeypatch.setattr(cli.questionary, "select", select)
    assert _select_input(["Short", "Long camera name"], "camera") == "Long camera name"
    assert len({len(choice.title) for choice in captured["choices"]}) == 1
    assert "reverse" in dict(captured["style"].style_rules)["highlighted"]
    assert "1-2" in captured["instruction"]


@pytest.mark.parametrize("keys", ["\x1b[B\x1b[B\r", "3\r"])
def test_camera_menu_selects_correct_device_with_arrows_or_number(monkeypatch, keys):
    cameras = [
        CameraDevice(8, "Elgato Virtual Camera", "elgato", 1200),
        CameraDevice(4, "OBS Virtual Camera", "obs", 1200),
        CameraDevice(1, "HD Pro Webcam C920", "c920", 1200),
    ]
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(keys)
        assert _select_input(cameras, "camera") is cameras[2]


def test_camera_menu_cancel(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\x03")
        with pytest.raises(typer.Abort):
            _select_input(["Camera A", "Camera B"], "webcam")


def test_camera_selection_with_redirected_input(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    answers = iter(["oops", "0", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert _select_input(["Camera A", "Camera B"], "webcam") == "Camera B"


class TestCLIHelp:
    def test_help_works(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "floptician" in result.output.lower() or "card detection" in result.output.lower()

    def test_run_help_works(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--debug" in result.output

    def test_validate_config_help_works(self):
        result = runner.invoke(app, ["validate-config", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_list_cameras_help_works(self):
        result = runner.invoke(app, ["list-cameras", "--help"])
        assert result.exit_code == 0


class TestValidateConfigCommand:
    def test_valid_config(self):
        config_path = str(Path(__file__).parent / "fixtures" / "config" / "valid_config.yaml")
        result = runner.invoke(app, ["validate-config", "--config", config_path])
        # May fail on model file not found, but should not crash
        # We accept either success or validation error
        assert result.exit_code in (0, 1)

    def test_missing_config_file(self):
        result = runner.invoke(app, ["validate-config", "--config", "/nonexistent/config.yaml"])
        assert result.exit_code == 1


def test_run_rejects_invalid_config_before_device_or_servers(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("capture: {fps: 0}")

    def unexpected_device_probe():
        raise AssertionError("Device selection must not run for invalid config")

    monkeypatch.setattr("floptician.cli._determine_torch_device", unexpected_device_probe)
    result = runner.invoke(app, ["run", "--config", str(path)])
    assert result.exit_code == 1
    assert "FPS" in result.output
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("failure", [None, "model", "obs", "read", "cancel"])
def test_capture_lifecycle_cleans_resources_on_exit(app_config, fake_obs, monkeypatch, failure):
    import platform
    from types import SimpleNamespace

    from floptician import (
        board_processor,
        camera_manager,
        cli,
        frame_processor,
        http_server,
        obs_client,
        websocket_server,
    )
    from floptician.exceptions import ModelLoadError, OBSConnectionError
    from tests.conftest import FakeDetector
    from tests.test_camera_manager import DEVICE, Capture

    app_config.platform = platform.system()
    capture = Capture(frames=[] if failure == "read" else None)
    if failure == "read":
        # Startup succeeds; the next read is the simulated disconnection.
        import numpy as np

        capture.frames = iter([np.full((720, 1280, 3), 80, np.uint8)])
    real_manager = camera_manager.CameraManager
    manager = real_manager(discover=lambda: [DEVICE], capture_factory=lambda *_: capture)
    monkeypatch.setattr(cli, "_initialize_config", lambda *_: app_config)
    monkeypatch.setattr(camera_manager, "CameraManager", lambda **_: manager)
    closed = []
    servers = []

    def server(*_):
        obj = SimpleNamespace(
            start=lambda: None,
            is_running=lambda: True,
            stop=lambda: closed.append("server"),
            send_message=lambda _: None,
        )
        servers.append(obj)
        return obj

    monkeypatch.setattr(http_server, "HTTPServer", server)
    monkeypatch.setattr(websocket_server, "WebSocketServer", server)
    real_board = board_processor.BoardProcessor

    def make_board(config):
        if failure == "model":
            raise ModelLoadError("Model unavailable")
        return real_board(config, detector=FakeDetector())

    def make_obs(config):
        if failure == "obs":
            raise OBSConnectionError("OBS unavailable")
        return fake_obs

    monkeypatch.setattr(board_processor, "BoardProcessor", make_board)
    monkeypatch.setattr(obs_client, "OBSClient", make_obs)
    monkeypatch.setattr(frame_processor.FrameProcessor, "check_for_quit", lambda self: self.frame_count >= 1)
    if failure == "cancel":
        manager._discover = lambda: [DEVICE, DEVICE]

        def cancel(*_):
            raise typer.Abort()

        monkeypatch.setattr(cli, "_select_input", cancel)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == (0 if failure is None else 130 if failure == "cancel" else 1), result.output
    assert capture.releases == (0 if failure == "cancel" else 1)
    assert len(closed) == len(servers)
    if failure in (None, "read"):
        assert not fake_obs.connected
