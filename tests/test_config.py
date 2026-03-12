from __future__ import annotations

import os
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

    def test_obs_password_default_empty(self):
        # Ensure OBS_PASSWORD env var is not set
        os.environ.pop("OBS_PASSWORD", None)
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
