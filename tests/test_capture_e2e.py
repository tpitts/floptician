"""OpenCV frames → real YOLO → stabilization → real WebSocket → served browser overlay."""

from __future__ import annotations

import json
import platform
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import pytest

from floptician.board_processor import BoardProcessor
from floptician.camera_devices import CameraDevice
from floptician.camera_manager import CameraManager
from floptician.exceptions import CameraError
from floptician.frame_processor import FrameProcessor
from floptician.http_server import HTTPServer
from floptician.models import AppConfig, CaptureConfig
from floptician.websocket_server import WebSocketServer
from tests import test_screenshot_pipeline

yolo_processor = test_screenshot_pipeline.yolo_processor


@pytest.mark.screenshot
def test_opencv_to_browser_board_appearance_removal_and_disconnect(tmp_path, fake_obs, yolo_processor):
    from playwright.sync_api import sync_playwright

    root = Path(__file__).parents[1]
    expected_path = root / "tests/screenshots/single_row/turn001.expected.json"
    expected = json.loads(expected_path.read_text())
    card_frame = cv2.imread(str(expected_path.with_name("turn001.png")))
    height, width = card_frame.shape[:2]
    # One startup frame, three consistent observations, six empty-table observations.
    for i in range(10):
        frame = card_frame.copy() if i < 4 else np.full_like(card_frame, 80)
        frame[0, 0] = (i + 1, 20, 30)  # Fresh frames, without altering any card.
        assert cv2.imwrite(str(tmp_path / f"frame{i:02d}.png"), frame)
    device = CameraDevice(0, "Recorded camera", "fixture", cv2.CAP_IMAGES)
    handles = []

    def open_images(index, backend):
        assert index == 0 and backend == cv2.CAP_IMAGES
        handle = cv2.VideoCapture(str(tmp_path / "frame%02d.png"), cv2.CAP_IMAGES)
        handles.append(handle)
        return handle

    config = AppConfig(platform=platform.system(), capture=CaptureConfig(width=width, height=height))
    now = [0.0]
    board = BoardProcessor(config, detector=yolo_processor, clock=lambda: now[0])
    with ExitStack() as resources:
        camera = resources.enter_context(CameraManager(discover=lambda: [device], capture_factory=open_images))
        camera.open_camera(device, camera_width=width, camera_height=height)
        config.capture.camera_manager = camera
        ws = WebSocketServer("127.0.0.1", 0)
        ws.start()
        resources.callback(ws.stop)
        port = ws.server.server_address[1]
        http = HTTPServer("127.0.0.1", 0, str(root / "src/floptician/static/overlay.html"), port)
        http.start()
        resources.callback(http.stop)
        assert http._is_running.wait(5)
        processor = FrameProcessor(config, fake_obs, board, ws)
        resources.callback(processor.shutdown)
        with sync_playwright() as p, p.chromium.launch(channel="chrome", headless=True) as browser:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
            page.route("https://fonts.gstatic.com/**", lambda route: route.abort())
            page.goto(f"http://127.0.0.1:{http.server.server_port}/")
            page.wait_for_function("socket.readyState === WebSocket.OPEN")

            def process_at(timestamp):
                now[0] = timestamp
                frame_info = processor.capture_frame()
                message = processor.process_frame(frame_info)
                assert message is not None
                ws.send_message(message)
                return message

            for timestamp in (0, 0.6):
                assert process_at(timestamp)["board"] == []
            message = process_at(1.25)
            expected_cards = sorted(c["card"] for c in expected["cards"])
            assert sorted(c["card"] for c in message["board"]) == expected_cards
            page.wait_for_function(
                "n => document.querySelectorAll('#board .card').length === n", arg=len(expected_cards)
            )
            assert (
                sorted(page.locator("#board .card").evaluate_all("els => els.map(e => e.dataset.card)"))
                == expected_cards
            )
            for timestamp in (2, 3, 4, 5, 6):
                assert process_at(timestamp)["board"]
            assert process_at(6.3)["board"] == []
            page.wait_for_function("document.querySelectorAll('#board .card').length === 0")
            with pytest.raises(CameraError, match="stopped delivering"):
                processor.capture_frame()
            assert not handles[0].isOpened()
            assert camera.get_frame() == (False, None)
            assert not errors
    assert not handles[0].isOpened()
    deadline = time.monotonic() + 2
    while (http.is_running() or ws.is_running()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not http.is_running() and not ws.is_running()
