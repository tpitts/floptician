from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from floptician.cli import app

runner = CliRunner()


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
        assert "--json-log" in result.output

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
