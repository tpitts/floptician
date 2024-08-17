import json
import threading
import time
import os
from datetime import datetime
import numpy as np
import yaml
import logging
import argparse
from aioconsole import ainput
from typing import Any, Dict
from app.custom_http_server import start_http_server
from app.custom_websocket_server import WebSocketServer
from app.board_processor import BoardProcessor, BoardState
from app.obs_client import OBSClient
from app.camera_manager import CameraManager
from app.frame_processor import FrameProcessor  # Import the new FrameProcessor


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
    with open(CONFIG_FILE, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Prefer the environment variable, fall back to config.yaml
obs_password = os.getenv('OBS_PASSWORD', config['obs']['password'])
config['obs']['password'] = obs_password

# Create output directory
OUTPUT_DIR = config['output_dir']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a unique folder for this run
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Floptician card detection system')
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
args = parser.parse_args()

# Set up logging
configure_logging(config.get('debug', False) or args.debug)

# Define the logger
logger = logging.getLogger(__name__)

websocket_server = WebSocketServer(config['host'], config['websocket_port'])

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

def start_capture_loop():
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
    except ValueError as e:
        logger.error(f"Error initializing BoardProcessor: {str(e)}")
        return

    try:
        # Connect to OBS WebSocket regardless of capture mode
        obs_client = OBSClient(config)
        logger.info(f"Connecting to OBS WebSocket at {config['obs']['host']}:{config['obs']['port']}")
        version_info = obs_client.get_version()
        logger.info(f"Connected to OBS WebSocket version: {version_info.obs_web_socket_version}")

        # Setup OBS overlay
        obs_client.setup_overlay()
        logger.info("OBS overlay setup complete")
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

# Start application
if __name__ == "__main__":
    try:
        # Start HTTP server in a new thread
        http_server_thread = threading.Thread(target=start_http_server, args=(config['host'], config['http_port'], config['html_file']))
        http_server_thread.daemon = True
        http_server_thread.start()
        logger.info("HTTP server thread started")

        # Start WebSocket server in a new thread
        websocket_server_thread = threading.Thread(target=websocket_server.start)
        websocket_server_thread.daemon = True
        websocket_server_thread.start()
        logger.info("WebSocket server thread started")

        # Give the servers a moment to start
        time.sleep(0.6)

        # Start capture loop
        start_capture_loop()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user, shutting down...")
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
    finally:
        logger.info("Exiting...")