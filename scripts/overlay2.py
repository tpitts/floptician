import json
import threading
import time
import os
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, HTTPServer
from websocket_server import WebsocketServer
import obsws_python as obs
import msvcrt
import base64
import numpy as np
import cv2
from PIL import Image


# Load connection configuration from JSON file
with open('connection_config.json', 'r') as config_file:
    config = json.load(config_file)
    HOST = config['host']
    OBS_WEBSOCKET_PORT = config['port']
    OBS_WEBSOCKET_PASSWORD = config['password']

# HTML file path
HTML_FILE_PATH = "overlay.html"

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create a unique folder for this run
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_output_dir = os.path.join(OUTPUT_DIR, run_timestamp)
os.makedirs(run_output_dir)

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
            with open(HTML_FILE_PATH, 'rb') as file:
                self.wfile.write(file.read())

# Function to start the HTTP server
def start_http_server():
    with HTTPServer((HOST, 8000), MyHandler) as httpd:
        print(f"Serving at port 8000")
        httpd.serve_forever()

# Function to start the WebSocket server
def start_websocket_server():
    server = WebsocketServer(host=HOST, port=9001)

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


def capture_webcam_frame(cl):
    try:
        # Get the list of video capture devices
        inputs = cl.get_input_list()
        video_sources = [input for input in inputs.inputs if input['inputKind'] == 'dshow_input']
        
        if not video_sources:
            print("No webcam sources found.")
            return
        
        # Use the first webcam source
        webcam_source = video_sources[0]['inputName']
        print(f"Using webcam source: {webcam_source}")
        
        # Hardcode 4:3 aspect ratio and 1280 width
        target_width = 1280
        target_height = int(target_width * (3/4))  # 4:3 aspect ratio
        
        print(f"Target resolution: {target_width}x{target_height}")
        
        # Capture the image at the target resolution
        response = cl.get_source_screenshot(
            webcam_source,
            "png",
            target_width,
            target_height,
            100  # quality (100 for best quality)
        )
        
        if not hasattr(response, 'image_data'):
            print("Error: 'image_data' not found in the response")
            return

        # The image_data starts with "data:image/png;base64,"
        base64_data = response.image_data.split(',', 1)[1]
        
        # Decode the base64 data
        image_data = base64.b64decode(base64_data)
        
        print(f"First 32 bytes of image data: {image_data[:32].hex()}")
        print(f"Total length of image data: {len(image_data)} bytes")

        # Save the image data
        image_path = os.path.join(run_output_dir, "webcam_frame.png")
        with open(image_path, "wb") as image_file:
            image_file.write(image_data)
        print(f"Image saved to: {image_path}")

        # Display the image using OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            height, width = img.shape[:2]
            print(f"Captured image dimensions: {width}x{height}")
            print(f"Aspect ratio: {width/height:.4f}")
            cv2.imshow('Captured Frame', img)
            cv2.waitKey(1)  # This will display the window and continue execution
            print("Image displayed in a window")
        else:
            print("Failed to decode image data")

    except Exception as e:
        print(f"Error capturing webcam frame: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()

def start_obs_websocket():
    cl = None
    try:
        cl = obs.ReqClient(host=HOST, port=OBS_WEBSOCKET_PORT, password=OBS_WEBSOCKET_PASSWORD)
        print(f"Connecting to OBS WebSocket at {HOST}:{OBS_WEBSOCKET_PORT}")

        # Get and print OBS WebSocket version
        version_info = cl.get_version()
        print(f"Connected to OBS WebSocket version: {version_info.obs_web_socket_version}")

        # Capture a frame from the webcam
        capture_webcam_frame(cl)

        print("Press 'q' to quit.")
        while True:
            if msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    print("Quitting...")
                    break
            time.sleep(0.1)  # Add a short delay to reduce CPU usage
    except Exception as e:
        print(f"Error in OBS WebSocket communication: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if cl:
            cl.disconnect()
        print("Disconnected from OBS WebSocket")

# Start HTTP server in a new thread
http_server_thread = threading.Thread(target=start_http_server)
http_server_thread.daemon = True
http_server_thread.start()

# Start WebSocket server in a new thread
websocket_server_thread = threading.Thread(target=start_websocket_server)
websocket_server_thread.daemon = True
websocket_server_thread.start()

# Start OBS WebSocket connection and capture frame
start_obs_websocket()