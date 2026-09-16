from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from floptician.config_utils import load_config
from floptician.models import DEFAULT_YOLO_MODEL

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("mac_configure", ROOT / "scripts/mac/configure.py")
configure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configure)


def test_existing_config_is_never_rewritten(tmp_path):
    path = tmp_path / "config.yaml"
    original = b"# My settings\nobs: {password: private}\nyolo: {model: custom.pt}\n"
    path.write_bytes(original)
    configure.create_config(tmp_path, ask=lambda _: pytest.fail("Should not prompt"))
    assert path.read_bytes() == original


def test_legacy_import_preserves_preferences_not_obsolete_paths(tmp_path):
    old = tmp_path / "old settings.yaml"
    old.write_text(
        yaml.safe_dump(
            {
                "obs": {"password": "private", "port": 4456},
                "html_file": "app/static/overlay.html",
                "host": "0.0.0.0",
                "capture": {"mode": "obs_websocket", "fps": 2.0, "width": 640},
                "yolo": {"model": "models/best.pt", "confidence_threshold": 0.8},
            }
        )
    )
    before = old.read_bytes()
    configure.create_config(tmp_path, ask=lambda _: str(old))
    data = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert data["obs"] == {"password": "private", "port": 4456}
    assert data["capture"] == {"mode": "direct_webcam", "fps": 2.0}
    assert data["yolo"] == {"model": "auto", "confidence_threshold": 0.8}
    assert "html_file" not in data and "host" not in data
    assert old.read_bytes() == before
    assert next((tmp_path / "output/setup-backups").glob("*/config.yaml")).read_bytes() == before
    assert (tmp_path / "config.yaml").stat().st_mode & 0o777 == 0o600


def test_new_setup_hides_password_and_uses_auto_model(tmp_path):
    configure.create_config(tmp_path, ask=lambda _: "", secret=lambda _: "a: secret # value")
    data = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert data["obs"]["password"] == "a: secret # value"
    assert data["yolo"]["model"] == "auto"


@pytest.mark.parametrize("system,machine,suffix", [("Darwin", "arm64", ".mlpackage"), ("Windows", "AMD64", ".pt")])
def test_auto_model_follows_platform_and_explicit_override_wins(tmp_path, monkeypatch, system, machine, suffix):
    monkeypatch.setattr("floptician.config_utils.platform.system", lambda: system)
    monkeypatch.setattr("floptician.config_utils.platform.machine", lambda: machine)
    path = tmp_path / "config.yaml"
    path.write_text("yolo: {model: auto}\n")
    assert Path(load_config(str(path)).yolo.model).name == Path(DEFAULT_YOLO_MODEL).with_suffix(suffix).name
    path.write_text("yolo: {model: custom.pt}\n")
    assert Path(load_config(str(path)).yolo.model) == tmp_path / "custom.pt"


@pytest.fixture
def managed_repo(tmp_path):
    root = tmp_path / "App with spaces"
    scripts = root / "scripts/mac"
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/mac/manage.sh", scripts / "manage.sh")
    for name in ("uv.lock", "pyproject.toml", ".python-version"):
        (root / name).write_text("test")
    tools = root / ".tools/bin"
    tools.mkdir(parents=True)
    # External tools are fake; exercise real shell sequencing without downloads,
    # touching the user's Python environment, OBS, or Git checkout.
    program = f"""#!{sys.executable}
import os, pathlib, sys
name=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
with open(os.environ["TRACE"], "a") as out: out.write(name+" "+" ".join(args)+"\\n")
if name == "uname": print("Darwin" if args == ["-s"] else "arm64")
elif name == "sw_vers": print(os.getenv("TEST_MACOS", "14.0"))
elif name == "df": print("disk 99999999 1 99999998")
elif name == "git":
    if args[:1] == ["status"]: print(os.getenv("TEST_DIRTY", ""))
    elif args[:1] == ["branch"]: print("master")
    elif args[:2] == ["remote", "get-url"]: print("https://github.com/tpitts/floptician")
    elif args[:1] == ["rev-parse"]: print("previous-revision")
    elif args[:1] == ["pull"] and os.getenv("TEST_PULL_FAIL"): sys.exit(1)
elif name == "uv":
    if args[0] == "sync":
        if os.getenv("TEST_SYNC_FAIL"): sys.exit(1)
        pathlib.Path(".venv").mkdir(exist_ok=True)
    elif "--model-include" in args: print("models/model.mlpackage/**")
"""
    for name in ("uname", "sw_vers", "xcode-select", "df", "git", "git-lfs", "uv"):
        path = tools / name
        path.write_text(program)
        path.chmod(0o755)
    trace = tmp_path / "trace"
    env = dict(os.environ, PATH=f"{tools}:/usr/bin:/bin", TRACE=str(trace))

    def run(action, **updates):
        result = subprocess.run(
            ["/bin/bash", str(scripts / "manage.sh"), action],
            env={**env, **updates},
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result, trace.read_text() if trace.exists() else ""

    return root, run


def test_setup_and_launch_with_spaces_and_locked_dependencies(managed_repo):
    root, run = managed_repo
    result, trace = run("setup")
    assert result.returncode == 0, result.stderr
    assert "uv sync --locked --no-dev --extra coreml" in trace
    assert "git lfs pull --include models/model.mlpackage/** --exclude" in trace
    assert (root / ".venv/floptician-mac-ready.txt").exists()
    assert not (root / ".floptician-operation.lock").exists()
    result, trace = run("run")
    assert result.returncode == 0, result.stderr
    assert "uv run --no-sync floptician run --config" in trace
    (root / "uv.lock").write_text("changed")
    result, _ = run("run")
    assert result.returncode != 0
    assert "dependencies changed" in result.stderr


def test_setup_refuses_unmanaged_environment_and_old_os(managed_repo):
    root, run = managed_repo
    (root / ".venv").mkdir()
    result, trace = run("setup")
    assert result.returncode != 0 and "unmanaged" in result.stderr
    assert "uv sync" not in trace
    result, _ = run("setup", TEST_MACOS="13.7")
    assert result.returncode != 0 and "macOS 14" in result.stderr


def test_failed_sync_is_retryable_and_cannot_launch(managed_repo):
    root, run = managed_repo
    result, _ = run("setup", TEST_SYNC_FAIL="1")
    assert result.returncode != 0
    assert not (root / ".venv/floptician-mac-ready.txt").exists()
    assert not (root / ".floptician-operation.lock").exists()
    assert run("run")[0].returncode != 0
    assert run("setup")[0].returncode == 0


def test_update_preserves_and_backs_up_config(managed_repo):
    root, run = managed_repo
    assert run("setup")[0].returncode == 0
    original = b"# keep comments and secrets\nobs: {password: private}\n"
    (root / "config.yaml").write_bytes(original)
    result, trace = run("update")
    assert result.returncode == 0, result.stderr
    assert "git pull --ff-only origin master" in trace
    assert (root / "config.yaml").read_bytes() == original
    backups = list((root / "output/setup-backups").glob("*/config.yaml"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def test_update_refuses_local_edits_and_run_lock(managed_repo):
    root, run = managed_repo
    result, trace = run("update", TEST_DIRTY=" M tracked.py")
    assert result.returncode != 0 and "local edits" in result.stderr
    assert "git pull" not in trace
    (root / ".floptician-operation.lock").mkdir()
    result, _ = run("update")
    assert result.returncode != 0 and "already running" in result.stderr


def test_failed_pull_does_not_sync_new_environment(managed_repo):
    _, run = managed_repo
    result, trace = run("update", TEST_PULL_FAIL="1")
    assert result.returncode != 0
    assert "uv sync" not in trace
