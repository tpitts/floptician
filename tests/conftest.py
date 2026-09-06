from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floptician.models import (
    AppConfig,
    BoardProcessorConfig,
    BoundingBox,
    CaptureConfig,
    CaptureMode,
    CardDetection,
    CommunityCard,
    OBSConfig,
    YOLOConfig,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIG_DIR = FIXTURES_DIR / "config"
IMAGES_DIR = FIXTURES_DIR / "images"


def pytest_addoption(parser):
    parser.addoption("--require-all-tests", action="store_true", help="Fail if any tests are skipped")


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--require-all-tests"):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter and reporter.stats.get("skipped"):
            reporter.write_sep("!", "Full validation requires zero skipped tests")
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def served_overlay():
    from floptician.http_server import HTTPServer

    servers = []

    def start(websocket_port=9001):
        html = Path(__file__).parents[1] / "src/floptician/static/overlay.html"
        server = HTTPServer("127.0.0.1", 0, str(html), websocket_port)
        servers.append(server)
        server.start()
        assert server._is_running.wait(5), "HTTP server did not start"
        return f"http://127.0.0.1:{server.server.server_port}/"

    yield start
    for server in servers:
        server.stop()
        server.server_thread.join(timeout=5)


# ── Fake implementations for protocols ──────────────────────────────────────────


class FakeCamera:
    """Implements CameraProtocol: returns frames from a list of numpy arrays."""

    def __init__(self, frames: list[np.ndarray] | None = None):
        self._frames = frames or []
        self._index = 0
        self.released = False

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return True, frame
        return False, None

    def release_camera(self) -> None:
        self.released = True


class FakeOBSClient:
    """Implements OBSProtocol: records calls, returns fake data."""

    def __init__(self):
        self.calls: list[str] = []
        self.connected = True

    def get_version(self):
        self.calls.append("get_version")

        class FakeVersion:
            obs_version = "30.0.0"
            obs_web_socket_version = "5.4.0"

        return FakeVersion()

    def get_webcams(self) -> list[str]:
        self.calls.append("get_webcams")
        return ["Logitech C920", "OBS Virtual Camera"]

    def capture_frame(self, webcam_source: str) -> bytes | None:
        self.calls.append(f"capture_frame:{webcam_source}")
        return None

    def setup_overlay(self) -> None:
        self.calls.append("setup_overlay")

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.connected = False


class FakeDetector:
    """Implements DetectorProtocol: returns preconfigured list[CardDetection]."""

    def __init__(self, detections: list[CardDetection] | None = None):
        self._detections = detections or []
        self.call_count = 0

    def process_frame(self, frame) -> list[CardDetection]:
        self.call_count += 1
        return self._detections


class FakeWebSocketServer:
    """Implements MessageBroadcaster: records sent messages."""

    def __init__(self):
        self.messages: list[dict] = []

    def send_message(self, message: dict, client=None) -> None:
        self.messages.append(message)


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """Minimal AppConfig for testing."""
    return AppConfig(
        debug=False,
        host="localhost",
        http_port=8000,
        websocket_port=9001,
        html_file="src/floptician/static/overlay.html",
        output_dir=str(tmp_path / "output"),
        platform="Windows",
        torch_device="cpu",
        config_path="",
        obs=OBSConfig(),
        capture=CaptureConfig(mode=CaptureMode.DIRECT_WEBCAM),
        yolo=YOLOConfig(model="models/fake.pt"),
        board_processor=BoardProcessorConfig(),
    )


@pytest.fixture
def board_processor_config() -> BoardProcessorConfig:
    return BoardProcessorConfig()


@pytest.fixture
def fake_obs() -> FakeOBSClient:
    return FakeOBSClient()


@pytest.fixture
def fake_ws() -> FakeWebSocketServer:
    return FakeWebSocketServer()


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A 640x360 black frame."""
    return np.zeros((360, 640, 3), dtype=np.uint8)


@pytest.fixture
def normal_frame() -> np.ndarray:
    """A 640x360 frame with varied pixel values (not black/white)."""
    rng = np.random.default_rng(42)
    return rng.integers(30, 200, size=(360, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[CardDetection]:
    """Three card detections in a row."""
    return [
        CardDetection(card="Ah", confidence=0.95, box=BoundingBox(100, 200, 160, 280)),
        CardDetection(card="Kd", confidence=0.92, box=BoundingBox(200, 200, 260, 280)),
        CardDetection(card="Qs", confidence=0.90, box=BoundingBox(300, 200, 360, 280)),
    ]


@pytest.fixture
def five_card_detections() -> list[CardDetection]:
    """Five card detections in a row."""
    return [
        CardDetection(card="Ah", confidence=0.95, box=BoundingBox(100, 200, 160, 280)),
        CardDetection(card="Kd", confidence=0.92, box=BoundingBox(200, 200, 260, 280)),
        CardDetection(card="Qs", confidence=0.90, box=BoundingBox(300, 200, 360, 280)),
        CardDetection(card="Jc", confidence=0.88, box=BoundingBox(400, 200, 460, 280)),
        CardDetection(card="Th", confidence=0.85, box=BoundingBox(500, 200, 560, 280)),
    ]


@pytest.fixture
def sample_community_cards() -> list[CommunityCard]:
    return [
        CommunityCard(card="Ah", x=1, y=1, confidence=0.95),
        CommunityCard(card="Kd", x=2, y=1, confidence=0.92),
        CommunityCard(card="Qs", x=3, y=1, confidence=0.90),
    ]
