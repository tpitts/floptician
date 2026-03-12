import argparse
import logging
import os
import platform
import sys
import time
from datetime import datetime

import torch

from floptician.board_processor import BoardProcessor
from floptician.camera_manager import CameraManager
from floptician.config_utils import load_config
from floptician.frame_processor import FrameProcessor
from floptician.http_server import HTTPServer
from floptician.models import AppConfig, CaptureMode
from floptician.obs_client import OBSClient
from floptician.websocket_server import WebSocketServer

logger = logging.getLogger(__name__)

BANNER = """
$$$$$$$$\\ $$\\                      $$\\     $$\\           $$\\
$$  _____|$$ |                     $$ |    \\__|          \\__|
$$ |      $$ | $$$$$$\\   $$$$$$\\ $$$$$$\\   $$\\  $$$$$$$\\ $$\\  $$$$$$\\  $$$$$$$\\
$$$$$\\    $$ |$$  __$$\\ $$  __$$\\\\_$$  _|  $$ |$$  _____|$$ | \\____$$\\ $$  __$$\\
$$  __|   $$ |$$ /  $$ |$$ /  $$ | $$ |    $$ |$$ /      $$ | $$$$$$$ |$$ |  $$ |
$$ |      $$ |$$ |  $$ |$$ |  $$ | $$ |$$\\ $$ |$$ |      $$ |$$  __$$ |$$ |  $$ |
$$ |      $$ |\\$$$$$$  |$$$$$$$  | \\$$$$  |$$ |\\$$$$$$$\\ $$ |\\$$$$$$$ |$$ |  $$ |
\\__|      \\__| \\______/ $$  ____/   \\____/ \\__| \\_______|\\__| \\_______|\\__|  \\__|
                        $$ |
                        $$ |
                        \\__|
"""


def determine_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif platform.system() == "Darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def configure_logging(debug_mode):
    logging_level = logging.DEBUG if debug_mode else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s", datefmt="%H:%M:%S")
    )
    root_logger.addHandler(stream_handler)

    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    logging.getLogger("obsws_python").setLevel(logging.ERROR)
    logging.getLogger("comtypes").setLevel(logging.ERROR)


def initialize_config() -> AppConfig:
    parser = argparse.ArgumentParser(description="Floptician card detection system")
    parser.add_argument("--config", help="Path to the config YAML file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)

    if args.debug:
        config.debug = True
    configure_logging(config.debug)

    config.platform = platform.system()
    config.torch_device = determine_torch_device()

    config.output_dir = os.path.join(config.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(config.output_dir, exist_ok=True)

    return config


def select_input(inputs, input_type):
    print(f"Available {input_type}s:")
    for i, input_device in enumerate(inputs):
        name = (
            input_device if isinstance(input_device, str) else input_device.get("camera_name", f"{input_type} {i + 1}")
        )
        print(f"{i + 1}. {name}")

    while True:
        selection = input(f"Enter the number of the {input_type} you want to use: ")
        try:
            index = int(selection) - 1
            if 0 <= index < len(inputs):
                return inputs[index]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def start_servers(config: AppConfig, timeout=5):
    http_server = None
    websocket_server = None
    try:
        http_server = HTTPServer(config.host, config.http_port, config.html_file)
        http_server.start()

        websocket_server = WebSocketServer(config.host, config.websocket_port)
        websocket_server.start()

        start_time = time.time()
        http_running = ws_running = False

        while time.time() - start_time < timeout:
            if not http_running and http_server.is_running():
                logger.info("HTTP server is running")
                http_running = True

            if not ws_running and websocket_server.is_running():
                logger.info("WebSocket server is running")
                ws_running = True

            if http_running and ws_running:
                break

            time.sleep(0.1)

        if not http_running:
            raise RuntimeError("HTTP server failed to start within the timeout period")
        if not ws_running:
            raise RuntimeError("WebSocket server failed to start within the timeout period")

        return http_server, websocket_server

    except Exception as e:
        logger.error(f"Error starting servers: {e}", exc_info=True)
        if http_server:
            http_server.stop()
        if websocket_server:
            websocket_server.stop()
        sys.exit(1)


def start_capture_loop(config: AppConfig, websocket_server):
    try:
        board_processor = BoardProcessor(config)
        logger.info(f"BoardProcessor started | YOLO model: {config.yolo.model} | Inference: {config.torch_device}")
    except ValueError as e:
        logger.error(f"Error initializing BoardProcessor: {e!s}")
        return

    try:
        obs_client = OBSClient(config)
        version_info = obs_client.get_version()
        logger.info(
            f"Connected to OBS WebSocket API {config.obs.host}:{config.obs.port} "
            f"using version: {version_info.obs_web_socket_version}"
        )
        obs_client.setup_overlay()
    except ConnectionRefusedError:
        logger.error(
            "Error: Unable to connect to OBS. Please ensure OBS is running and the WebSocket server is enabled."
        )
        return
    except Exception as e:
        logger.error(f"Error: Unexpected issue when connecting to OBS: {e!s}")
        return

    if config.capture.mode == CaptureMode.OBS_WEBSOCKET:
        webcams = obs_client.get_webcams()
        if not webcams:
            logger.error("No webcams found in OBS. Exiting...")
            return
        config.capture.selected_webcam = webcams[0] if len(webcams) == 1 else select_input(webcams, "webcam")

    elif config.capture.mode == CaptureMode.DIRECT_WEBCAM:
        camera_manager = CameraManager()
        cameras = camera_manager.get_available_cameras()
        if not cameras:
            logger.error("No webcams found. Exiting...")
            return

        selected_camera = cameras[0] if len(cameras) == 1 else select_input(cameras, "camera")
        if not camera_manager.open_camera(selected_camera["camera_index"]):
            logger.error(f"Failed to open camera {selected_camera['camera_name']}. Exiting...")
            return

        camera_manager.set_resolution(config.capture.width, config.capture.height)
        config.capture.camera_manager = camera_manager

    else:
        logger.error(f"Invalid capture mode: {config.capture.mode}. Exiting...")
        return

    logger.info("Starting frame processor...")
    frame_processor = FrameProcessor(config, obs_client, board_processor, websocket_server)

    try:
        frame_processor.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in start_capture_loop: {e!s}", exc_info=True)
    finally:
        logger.info("Exiting capture loop")


def main():
    print(BANNER)

    config = initialize_config()

    if config.debug:
        logger.debug("Debug mode is enabled.")
    logger.info(f"Platform: {config.platform}")
    logger.info(f"Inference Device: {config.torch_device}")

    http_server, websocket_server = start_servers(config)

    start_capture_loop(config, websocket_server)

    http_server.stop()
    websocket_server.stop()


if __name__ == "__main__":
    main()
