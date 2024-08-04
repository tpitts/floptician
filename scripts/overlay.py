import json
import threading
import time
import os
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, HTTPServer
from websocket_server import WebsocketServer
import msvcrt
import base64
import numpy as np
import cv2
from PIL import Image
import yaml
from ultralytics import YOLO

from obs_client import OBSClient

# Configuration
CONFIG_FILE = 'config.yaml'

def load_config():
    with open(CONFIG_FILE, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Create output directory
OUTPUT_DIR = config['output_dir']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a unique folder for this run
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

# HTTP Server Handler
class MyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ws":
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            with open(config['html_file'], 'rb') as file:
                self.wfile.write(file.read())

# Function to start the HTTP server
def start_http_server():
    try:
        with HTTPServer((config['host'], config['http_port']), MyHandler) as httpd:
            print(f"HTTP Server started at http://{config['host']}:{config['http_port']}")
            httpd.serve_forever()
    except Exception as e:
        print(f"Error starting HTTP server: {e}")

# WebSocket server
def start_websocket_server():
    server = WebsocketServer(host=config['host'], port=config['websocket_port'])

    def new_client(client, server):
        print(f"New client connected and was given id {client['id']}")
        server.send_message(client, "show")

    def client_left(client, server):
        print(f"Client({client['id']}) disconnected")
        server.send_message_to_all("hide")

    def message_received(client, server, message):
        print(f"Client({client['id']}) said: {message}")

    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    server.run_forever()

# YOLO Model
class YOLOModel:
    def __init__(self):
        self.model = YOLO(config['yolo']['model'])

    def process_image(self, image_data):
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = self.model(img)
        
        # Extract class labels from results
        detected_classes = [result.names[int(cls)] for result in results for cls in result.boxes.cls]
        
        # Count occurrences of each class
        class_counts = {}
        for cls in detected_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Format the results
        formatted_results = [f"{cls}: {count}" for cls, count in class_counts.items()]
        
        return {"detected_objects": formatted_results}

# Main application
def start_obs_websocket():
    obs_client = OBSClient(config)
    yolo_model = YOLOModel()

    print(f"Connecting to OBS WebSocket at {config['obs']['host']}:{config['obs']['port']}")
    version_info = obs_client.get_version()
    print(f"Connected to OBS WebSocket version: {version_info.obs_web_socket_version}")

    obs_client.setup_overlay()

    print("Press 'q' to quit.")
    while True:
        image_data = obs_client.capture_frame()
        if image_data:
            print("Frame captured, processing with YOLO model...")
            results = yolo_model.process_image(image_data)
            print(f"YOLO results: {results}")
            # Here you would send the results to the WebSocket server
            # For now, we'll just print them
            print("Results:", results)
        else:
            print("Failed to capture frame")
        
        if msvcrt.kbhit():
            if msvcrt.getch().lower() == b'q':
                print("Quitting...")
                break
        time.sleep(1 / config['capture']['fps'])

    obs_client.disconnect()
    print("Disconnected from OBS WebSocket")

# Start application
if __name__ == "__main__":
    # Start HTTP server in a new thread
    http_server_thread = threading.Thread(target=start_http_server)
    http_server_thread.daemon = True
    http_server_thread.start()
    print("HTTP server thread started")

    # Start WebSocket server in a new thread
    websocket_server_thread = threading.Thread(target=start_websocket_server)
    websocket_server_thread.daemon = True
    websocket_server_thread.start()
    print("WebSocket server thread started")

    # Give the servers a moment to start
    time.sleep(2)

    # Start OBS WebSocket connection and manage overlay
    try:
        start_obs_websocket()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
    finally:
        print("Exiting...")