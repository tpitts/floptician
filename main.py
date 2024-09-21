import platform
import threading
import time
import os
import sys
from datetime import datetime
import numpy as np
import yaml
import logging
import argparse
import torch
from aioconsole import ainput
from typing import Any, Dict
from app.custom_http_server import HTTPServer
from app.custom_websocket_server import WebSocketServer
from app.board_processor import BoardProcessor, BoardState
from app.obs_client import OBSClient
from app.camera_manager import CameraManager
from app.frame_processor import FrameProcessor

# Create logger after configuring
logger = logging.getLogger(__name__)

text = """
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

print(text)

# Configuration
CONFIG_FILE = 'config.yaml'

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file)
        if not config:
            raise ValueError("Config file is empty")
        return config
    except FileNotFoundError:
        print(f"Config file {CONFIG_FILE} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

def determine_torch_device() -> str:
    """
    Determine the best available PyTorch device for YOLO inference.

    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif (platform.system() == "Darwin" and
          platform.machine() == "arm64" and
          torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'

# Set up logging configuration
def configure_logging(debug_mode):
    logging_level = logging.DEBUG if debug_mode else logging.INFO

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    
    # Remove any existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Add stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(stream_handler)

    # Suppress YOLO logging
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    logging.getLogger('obsws_python').setLevel(logging.ERROR)
    logging.getLogger('comtypes').setLevel(logging.ERROR)

def initialize_config() -> Dict[str, Any]:
    """
    Initialize and return the configuration for the application.

    Returns:
        Dict[str, Any]: Initialized configuration.
    """
    parser = argparse.ArgumentParser(description='Floptician card detection system')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    config = load_config()

    config['debug'] = args.debug or config.get('debug', False)
    configure_logging(config['debug'])

    config['platform'] = platform.system()
    config['torch_device'] = determine_torch_device()

    #if config['torch_device'] == 'mps':
    #    config['yolo']['model'] = config['yolo'].get('coreml_model', config['yolo']['model'])

    config['obs']['password'] = os.getenv('OBS_PASSWORD', config['obs']['password'])

    config['output_dir'] = os.path.join(config['output_dir'], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(config['output_dir'], exist_ok=True)

    return config

def select_input(inputs, input_type):
    print(f"Available {input_type}s:")
    for i, input_device in enumerate(inputs):
        name = input_device if isinstance(input_device, str) else input_device.get('camera_name', f"{input_type} {i+1}")
        print(f"{i+1}. {name}")
    
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

def start_servers(config, timeout=5):
    """Start HTTP and WebSocket servers and ensure they are ready."""
    http_server = None
    websocket_server = None
    try:
        http_server = HTTPServer(config['host'], config['http_port'], config['html_file'])
        http_server.start()

        websocket_server = WebSocketServer(config['host'], config['websocket_port'])
        websocket_server.start()

        start_time = time.time()
        http_running = ws_running = False

        # Loop until the servers are running or the timeout is reached
        while time.time() - start_time < timeout:
            if not http_running and http_server.is_running():
                logger.info("HTTP server is running")
                http_running = True
            
            if not ws_running and websocket_server.is_running():
                logger.info("WebSocket server is running")
                ws_running = True

            if http_running and ws_running:
                break  # Exit loop if both servers are confirmed running
            
            time.sleep(0.1)  # Small sleep to avoid busy-waiting

        # If either server did not start within the timeout, raise an error
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

def start_capture_loop(config, websocket_server):
    if 'capture' not in config:
        logger.error("Error: 'capture' section missing from configuration. Please check your config.yaml file.")
        return
    if 'mode' not in config['capture']:
        logger.error("Error: 'mode' not specified in capture configuration. Please check your config.yaml file.")
        return
    if 'yolo' not in config or 'model' not in config['yolo']:
        logger.error("Error: YOLO model path not found in configuration. Please check your config.yaml file.")
        return

    try:
        board_processor = BoardProcessor(config)
        logger.info(f"BoardProcessor started | YOLO model: {config['yolo']['model']} | Inference: {config['torch_device']}")
    except ValueError as e:
        logger.error(f"Error initializing BoardProcessor: {str(e)}")
        return

    try:
        # Connect to OBS WebSocket regardless of capture mode
        obs_client = OBSClient(config)
        version_info = obs_client.get_version()
        logger.info(f"Connected to OBS WebSocket API {config['obs']['host']}:{config['obs']['port']} "
            f"using version: {version_info.obs_web_socket_version}")

        # Setup OBS overlay
        obs_client.setup_overlay()

    except ConnectionRefusedError:
        logger.error("Error: Unable to connect to OBS. Please ensure OBS is running and the WebSocket server is enabled.")
        return
    except Exception as e:
        logger.error(f"Error: Unexpected issue when connecting to OBS: {str(e)}")
        return

    # Set up capture method based on config
    if config['capture']['mode'] == 'obs_websocket':
        webcams = obs_client.get_webcams()
        if not webcams:
            logger.error("No webcams found in OBS. Exiting...")
            return

        config['capture']['selected_webcam'] = webcams[0] if len(webcams) == 1 else select_input(webcams, "webcam")

    elif config['capture']['mode'] == 'direct_webcam':
        camera_manager = CameraManager()
        cameras = camera_manager.get_available_cameras()
        if not cameras:
            logger.error("No webcams found. Exiting...")
            return

        selected_camera = cameras[0] if len(cameras) == 1 else select_input(cameras, "camera")
        if not camera_manager.open_camera(selected_camera['camera_index']):
            logger.error(f"Failed to open camera {selected_camera['camera_name']}. Exiting...")
            return

        camera_manager.set_resolution(config['capture']['width'], config['capture']['height'])
        config['capture']['camera_manager'] = camera_manager

    else:
        logger.error(f"Invalid capture mode: {config['capture']['mode']}. Exiting...")
        return
    
    logger.info("Starting frame processor...")
    frame_processor = FrameProcessor(config, obs_client, board_processor, websocket_server)
    
    try:
        frame_processor.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in start_capture_loop: {str(e)}", exc_info=True)
    finally:
        logger.info("Exiting capture loop")

def main():
    config = initialize_config()
    
    if config['debug']:
        logger.debug("Debug mode is enabled.")
    logger.info(f"Platform: {config['platform']}")
    logger.info(f"Inference Device: {config['torch_device']}")

    # Start servers and retrieve both HTTP and WebSocket server instances
    http_server, websocket_server = start_servers(config)

    start_capture_loop(config, websocket_server)

    http_server.stop()
    websocket_server.stop()

if __name__ == "__main__":
    main()
