from __future__ import annotations

from pathlib import Path

import pytest

from floptician.config_utils import find_config_path, load_config, validate_config
from floptician.exceptions import ConfigurationError
from floptician.models import AppConfig, CaptureMode

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "config"


class TestLoadConfig:
    def test_valid_config_returns_app_config(self):
        config = load_config(str(FIXTURES_DIR / "valid_config.yaml"))
        assert isinstance(config, AppConfig)
        assert config.host == "localhost"
        assert config.http_port == 8000
        assert config.capture.mode == CaptureMode.DIRECT_WEBCAM
        assert config.capture.fps == 1.8
        assert config.yolo.image_size == 1280
        assert config.yolo.coreml_compute_unit == "cpu-and-ne"
        assert config.yolo.confidence_threshold == 0.70
        assert config.board_processor.min_frames_to_show == 3

    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_empty_config_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ConfigurationError, match="empty"):
            load_config(str(empty))

    def test_invalid_yaml_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo: [bar: baz")
        with pytest.raises(ConfigurationError, match="Error parsing"):
            load_config(str(bad))

    def test_obs_password_env_override(self, monkeypatch):
        monkeypatch.setenv("OBS_PASSWORD", "secret123")
        config = load_config(str(FIXTURES_DIR / "valid_config.yaml"))
        assert config.obs.password == "secret123"

    def test_obs_password_default_empty(self, monkeypatch):
        monkeypatch.delenv("OBS_PASSWORD", raising=False)
        config = load_config(str(FIXTURES_DIR / "valid_config.yaml"))
        assert config.obs.password == ""


class TestValidateConfig:
    def test_valid_config_passes(self):
        load_config(str(FIXTURES_DIR / "valid_config.yaml"))
        # Should not raise — validates config structure loads correctly

    def test_invalid_confidence_threshold(self, app_config: AppConfig):
        app_config.yolo.confidence_threshold = 5.0
        with pytest.raises(ConfigurationError, match="confidence_threshold"):
            validate_config(app_config)

    def test_invalid_overlap_threshold(self, app_config: AppConfig):
        app_config.yolo.overlap_threshold = -1.0
        with pytest.raises(ConfigurationError, match="overlap_threshold"):
            validate_config(app_config)

    def test_negative_fps_raises(self, app_config: AppConfig, tmp_path: Path):
        model = tmp_path / "fake_model.pt"
        model.write_text("fake")
        app_config.yolo.model = str(model)
        app_config.capture.fps = -1.0
        with pytest.raises(ConfigurationError, match="FPS"):
            validate_config(app_config)

    def test_missing_model_file_raises(self, app_config: AppConfig):
        app_config.yolo.model = "/nonexistent/model.pt"
        with pytest.raises(ConfigurationError, match="model file not found"):
            validate_config(app_config)


class TestFindConfigPath:
    def test_explicit_path_works(self, tmp_path: Path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("debug: true")
        assert find_config_path(str(cfg)) == cfg

    def test_explicit_path_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            find_config_path("/nonexistent/config.yaml")

    def test_env_var_override(self, monkeypatch, tmp_path: Path):
        cfg = tmp_path / "env_config.yaml"
        cfg.write_text("debug: true")
        monkeypatch.setenv("FLOPTICIAN_CONFIG", str(cfg))
        assert find_config_path() == cfg


@pytest.mark.parametrize(
    "content", ["[]", "true", "capture: []", "obs: null", "capture: {mode: wrong}", "yolo: {model: 123}"]
)
def test_malformed_settings_have_clear_errors(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        load_config(str(path))


@pytest.mark.parametrize(
    "section,name,value",
    [
        ("capture", "fps", "fast"),
        ("capture", "fps", float("nan")),
        ("capture", "width", 0),
        ("capture", "height", True),
        (None, "http_port", 65536),
        (None, "websocket_port", "9001"),
        ("obs", "port", -1),
        ("yolo", "confidence_threshold", True),
        ("yolo", "image_size", 1000),
        ("board_processor", "min_frames_to_show", 0),
        ("board_processor", "min_frames_to_remove", 1.5),
        ("board_processor", "min_ms_to_remove", -1),
        ("board_processor", "vertical_alignment_threshold", 0),
    ],
)
def test_invalid_numeric_settings(app_config, section, name, value):
    target = getattr(app_config, section) if section else app_config
    setattr(target, name, value)
    with pytest.raises(ConfigurationError, match=name if name != "fps" else "FPS"):
        validate_config(app_config)


def test_ports_must_be_distinct(app_config):
    app_config.websocket_port = app_config.http_port
    with pytest.raises(ConfigurationError, match="must be different"):
        validate_config(app_config)


def test_coreml_compute_unit_must_be_supported(app_config):
    app_config.yolo.coreml_compute_unit = "gpu-only"
    with pytest.raises(ConfigurationError, match="coreml_compute_unit"):
        validate_config(app_config)


def test_valid_config_passes_full_validation(app_config, tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"placeholder")
    app_config.yolo.model = str(model)
    validate_config(app_config)
