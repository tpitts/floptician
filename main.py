import json
import threading
import time
import os
from datetime import datetime
import numpy as np
import cv2
import yaml
import logging
import argparse
from app.custom_http_server import start_http_server
from app.custom_websocket_server import WebSocketServer
from app.board_processor import BoardProcessor
from app.obs_client import OBSClient
from app.camera_manager import CameraManager

#windows specific, needs a fix
import msvcrt

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

# Set up logging
logger = logging.getLogger('echelon')
logger.setLevel(logging.INFO)  # Default to INFO level

# Create console handler and set level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

# Prevent the logger from propagating to the root logger
logger.propagate = False

# Parse command line arguments
parser = argparse.ArgumentParser(description='Echelon card detection system')
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
args = parser.parse_args()

if config['debug'] or args.debug:
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)

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
        board_processor = BoardProcessor(config['yolo']['model'])
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

        selected_webcam = webcams[0] if len(webcams) == 1 else select_input(webcams, "webcam")
        capture_frame = lambda: obs_client.capture_frame(selected_webcam)

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
        capture_frame = camera_manager.get_frame

    else:
        logger.error(f"Invalid capture mode: {config['capture']['mode']}. Exiting...")
        return

    logger.info("Press 'q' to quit.")
    frame_count = 0
    total_time = 0

    while True:
        try:
            loop_start = time.time()

            # Capture frame
            if config['capture']['mode'] == 'obs_websocket':
                image_data = capture_frame()
                if image_data:
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    frame = None
            else:
                success, frame = capture_frame()
                if not success:
                    frame = None

            if frame is not None:
                # Board processing
                try:
                    result = board_processor.process_frame(frame)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    continue

                # Add frame_count to the result
                result['frame_count'] = frame_count

                # WebSocket update (send for every frame)
                websocket_server.send_message(result)

                # Logging
                if result['state'] == 'stable':
                    logger.info(f"Stable board detected: {json.dumps(result['board'], indent=2)}")
                elif result['state'] == 'detected':
                    logger.info("Unstable board detected")
                else:
                    logger.info("No board detected")

                logger.debug(f"Frame {frame_count} processed: {json.dumps(result, indent=2)}")

                frame_count += 1
                loop_duration = time.time() - loop_start
                total_time += loop_duration

                # Performance logging
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames. Average processing time: {total_time/frame_count:.4f}s")

            else:
                logger.warning("Failed to capture frame")

            if msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    logger.info("Quitting...")
                    break

            # Calculate sleep time to maintain desired FPS
            sleep_time = max(0, (1 / config['capture']['fps']) - loop_duration)
            time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")

    if config['capture']['mode'] == 'obs_websocket':
        obs_client.disconnect()
    else:
        camera_manager.release_camera()

    logger.info("Disconnected from capture source")
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Average processing time: {total_time/frame_count:.4f}s")
    logger.info(f"Effective FPS: {frame_count/total_time:.2f}")

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
        time.sleep(2)

        # Start capture loop
        start_capture_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down...")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
    finally:
        print("Exiting...")