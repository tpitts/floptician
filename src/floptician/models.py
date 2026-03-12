from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

# ── Enums ──────────────────────────────────────────────────────────────────────


class BoardState(Enum):
    NOT_SHOWING = "Not Showing"
    SHOWING = "Showing"


class TransitionState(Enum):
    NONE = "None"
    APPEARING = "Appearing"
    DISAPPEARING = "Disappearing"


class BoardConfiguration(Enum):
    NO_BOARD = auto()
    SINGLE_ROW = auto()
    TWO_ROWS = auto()
    CHIHUAHUA = auto()


class CaptureMode(Enum):
    OBS_WEBSOCKET = "obs_websocket"
    DIRECT_WEBCAM = "direct_webcam"


class FrameProcessorState(Enum):
    RUNNING = 1
    FAILED = 2


# ── Config dataclasses ─────────────────────────────────────────────────────────


@dataclass
class OBSConfig:
    host: str = "localhost"
    port: int = 4455
    password: str = ""


@dataclass
class CaptureConfig:
    mode: CaptureMode = CaptureMode.DIRECT_WEBCAM
    width: int = 640
    height: int = 360
    fps: float = 1.8
    # Runtime state (set after init, not from config file)
    selected_webcam: str | None = None
    camera_manager: object | None = None


@dataclass
class YOLOConfig:
    model: str = "models/yolov8l-2026-03-10.pt"
    confidence_threshold: float = 0.70
    overlap_threshold: float = 0.80


@dataclass
class BoardProcessorConfig:
    vertical_alignment_threshold: float = 0.20
    horizontal_alignment_threshold: float = 0.10
    image_height: int = 1080
    min_frames_to_show: int = 3
    min_ms_to_show: int = 1200
    min_frames_to_remove: int = 6
    min_ms_to_remove: int = 4200


@dataclass
class AppConfig:
    debug: bool = False
    host: str = "localhost"
    http_port: int = 8000
    websocket_port: int = 9001
    html_file: str = "src/floptician/static/overlay.html"
    output_dir: str = "output"
    platform: str = ""
    torch_device: str = "cpu"
    config_path: str = ""

    obs: OBSConfig = field(default_factory=OBSConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    board_processor: BoardProcessorConfig = field(default_factory=BoardProcessorConfig)


# ── Domain dataclasses ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: list[float]) -> BoundingBox:
        return cls(coords[0], coords[1], coords[2], coords[3])


@dataclass
class CardDetection:
    card: str
    confidence: float
    box: BoundingBox


@dataclass
class CommunityCard:
    card: str
    x: int
    y: int
    confidence: float = 1.0


@dataclass
class FrameInfo:
    frame_id: int
    frame: object  # numpy ndarray
    capture_time: float


@dataclass
class BoardResult:
    timestamp: float
    state: BoardState
    board: list[CommunityCard]
    debug_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state": self.state.value,
            "board": [{"card": c.card, "x": c.x, "y": c.y, "confidence": c.confidence} for c in self.board],
            "debug_info": self.debug_info,
        }
