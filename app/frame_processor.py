import time
import logging
from dataclasses import dataclass
from typing import Any, Dict
import cv2
import numpy as np
from enum import Enum
from app.board_processor import BoardProcessor, BoardState
from app.obs_client import OBSClient
from app.camera_manager import CameraManager
from app.custom_websocket_server import WebSocketServer
import sys
import os

logger = logging.getLogger(__name__)

@dataclass
class FrameInfo:
    frame_id: int
    frame: Any
    capture_time: float

class FrameProcessor:
    """
    Manages the capture, processing, and distribution of video frames.
    Runs operations serially for simplicity.
    """

    def __init__(self, config, obs_client, board_processor, websocket_server):
        # Initialize with configuration and necessary components
        self.config = config
        self.obs_client = obs_client
        self.board_processor = board_processor
        self.websocket_server = websocket_server
        
        # Set up frame rate control
        self.target_fps = config['capture']['fps']
        self.frame_interval = 1 / self.target_fps
        
        # Initialize frame tracking
        self.last_frame_id = 0
        self.last_processed_id = 0
        
        # Set up runtime control
        self.running = True
        
        # Initialize performance tracking
        self.total_time = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Set up platform-specific input handling
        if os.name == 'nt':  # Windows
            import msvcrt
            self.kbhit = msvcrt.kbhit
            self.getch = msvcrt.getch
        else:  # Unix-like
            import termios
            import fcntl
            import tty
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            self.setup_unix_input()

    def setup_unix_input(self):
        """Set up non-blocking input for Unix-like systems."""
        new_settings = termios.tcgetattr(self.fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSANOW, new_settings)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, fcntl.fcntl(self.fd, fcntl.F_GETFL) | os.O_NONBLOCK)

    def capture_frame(self) -> FrameInfo:
        """
        Capture a frame from the configured source (OBS or direct camera).
        """
        try:
            if self.config['capture']['mode'] == 'obs_websocket':
                # Capture from OBS
                image_data = self.obs_client.capture_frame(self.config['capture']['selected_webcam'])
                if image_data:
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    frame = None
            else:
                # Capture from direct camera
                success, frame = self.config['capture']['camera_manager'].get_frame()
                if not success:
                    frame = None

            self.last_frame_id += 1
            return FrameInfo(self.last_frame_id, frame, time.time())
        except Exception as e:
            logger.error(f"Error in frame capture: {str(e)}", exc_info=True)
            return None

    def process_frame(self, frame_info: FrameInfo) -> Dict[str, Any]:
        """
        Process a captured frame using the board processor.
        """
        try:
            start_time = time.time()
            result = self.board_processor.process_frame(frame_info.frame)
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self.total_time += processing_time
            self.frame_count += 1

            # Log results based on board state
            if result['state'] == BoardState.SHOWING:
                # Sort the cards first by y, then by x
                sorted_board = sorted(result['board'], key=lambda card: (card['y'], card['x']))
                # Extract only the card values in the sorted order
                card_values = [card['card'] for card in sorted_board]
                logger.debug(f"Stable board: {card_values}")
            elif result['state'] == BoardState.NOT_SHOWING:
                logger.debug("No board.")
            else:
                logger.warning(f"Unexpected board state: {result['state']}")

            # Prepare result for transmission
            result = self.convert_enums_to_strings(result)
            result['frame_id'] = frame_info.frame_id
            result['frame_count'] = self.frame_count
            result['processing_time'] = processing_time

            logger.debug(f"Frame {result['frame_id']} | Processing time: {result['processing_time']:.3f}s")

            # Log performance metrics periodically
            if self.frame_count % 100 == 0:
                logger.info(f"Processed {self.frame_count} frames. Average processing time: {self.total_time/self.frame_count:.4f}s")

            return result
        except Exception as e:
            logger.error(f"Error in frame processing: {str(e)}", exc_info=True)
            return None

    def convert_enums_to_strings(self, obj):
        """
        Recursively convert Enum values to strings for JSON serialization.
        """
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: self.convert_enums_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_enums_to_strings(item) for item in obj]
        return obj

    def run(self):
        """
        Main run method to start the frame processing loop.
        Handles exceptions and performs cleanup on exit.
        """
        try:
            while self.running:
                frame_info = self.capture_frame()
                if frame_info and frame_info.frame is not None:
                    if frame_info.frame_id > self.last_processed_id:
                        result = self.process_frame(frame_info)
                        if result:
                            self.websocket_server.send_message(result)
                            self.last_processed_id = frame_info.frame_id
                    else:
                        logger.info(f"Dropping old frame {frame_info.frame_id}")
                
                # Maintain target frame rate
                time.sleep(self.frame_interval)

                # Check for quit command
                if self.check_for_quit():
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}", exc_info=True)
        finally:
            self.shutdown()

    def check_for_quit(self):
        """
        Check for user input to quit the application without blocking.
        """
        if os.name == 'nt':  # Windows
            if self.kbhit():
                if self.getch() == b'q':
                    logger.info("Quit command received. Shutting down...")
                    return True
        else:  # Unix-like
            try:
                c = sys.stdin.read(1)
                if c == 'q':
                    logger.info("Quit command received. Shutting down...")
                    return True
            except IOError:
                pass
        return False

    def shutdown(self):
        """
        Gracefully shut down and release resources.
        """
        logger.info("Initiating shutdown...")
        self.running = False
        if self.config['capture']['mode'] == 'obs_websocket':
            self.obs_client.disconnect()
        elif hasattr(self.config['capture'], 'camera_manager'):
            self.config['capture']['camera_manager'].release_camera()
        
        if os.name != 'nt':
            # Restore terminal settings for Unix-like systems
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_settings)
        
        end_time = time.time()
        total_run_time = end_time - self.start_time
        logger.info(f"Total run time: {total_run_time:.2f}s")
        logger.info(f"Total frames processed: {self.frame_count}")
        if self.frame_count > 0:
            logger.info(f"Average processing time: {self.total_time/self.frame_count:.4f}s")
            logger.info(f"Effective FPS: {self.frame_count/total_run_time:.2f}")
        logger.info("Shutdown complete")