import time
import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum
from app.board_processor import BoardProcessor, BoardState
import sys
import os

logger = logging.getLogger(__name__)

class FrameProcessorState(Enum):
    RUNNING = 1
    FAILED = 2

@dataclass
class FrameInfo:
    frame_id: int
    frame: Any
    capture_time: float

class FrameProcessor:
    def __init__(self, config, obs_client, board_processor, websocket_server):
        self.config = config
        self.obs_client = obs_client
        self.board_processor = board_processor
        self.websocket_server = websocket_server
        
        self.target_fps = config['capture']['fps']
        self.frame_interval = 1 / self.target_fps
        
        self.last_frame_id = 0
        self.last_processed_id = 0
        
        self.total_processing_time = 0
        self.frame_count = 0
        
        self.state = FrameProcessorState.RUNNING
        self.running = True
        
        self.start_time = time.time()
        self.last_valid_frame_time = self.start_time
        
        self.first_frame_threshold = 8.0  # 8 seconds threshold for the first frame
        self.subsequent_frame_threshold = 4.0  # 4 seconds threshold for subsequent frames
        self.is_first_frame = True
        
        self.debug_mode = config.get('debug', False)
        
        self.previous_frame = None

        if config['platform'] == 'Windows':
            import msvcrt
            self.kbhit = msvcrt.kbhit
            self.getch = msvcrt.getch
        else:
            import termios
            import fcntl
            import tty
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            self.setup_unix_input()

    def setup_unix_input(self):
        new_settings = termios.tcgetattr(self.fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSANOW, new_settings)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, fcntl.fcntl(self.fd, fcntl.F_GETFL) | os.O_NONBLOCK)

    def capture_frame(self) -> Optional[FrameInfo]:
        try:
            if self.config['capture']['mode'] == 'obs_websocket':
                image_data = self.obs_client.capture_frame(self.config['capture']['selected_webcam'])
                if image_data:
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    frame = None
            else:
                success, frame = self.config['capture']['camera_manager'].get_frame()
                if not success:
                    frame = None

            self.last_frame_id += 1
            return FrameInfo(self.last_frame_id, frame, time.time()) if frame is not None else None
        except Exception as e:
            logger.error(f"Error in frame capture: {str(e)}", exc_info=True)
            return None

    def is_valid_frame(self, frame: np.ndarray) -> bool:
        if frame is None:
            logger.warning("No frame captured")
            return False
        
        # Check for 50% black or white pixels
        total_pixels = frame.shape[0] * frame.shape[1]
        black_pixels = np.sum(frame == 0) / 3  # Divide by 3 for RGB channels
        white_pixels = np.sum(frame == 255) / 3
        
        if black_pixels / total_pixels > 0.5:
            logger.warning("Frame is over 50% black")
            return False
        
        if white_pixels / total_pixels > 0.5:
            logger.warning("Frame is over 50% white")
            return False
        
        # Check for frozen frame
        if self.previous_frame is not None:
            if np.array_equal(frame, self.previous_frame):
                logger.warning("Frozen frame detected")
                return False
        
        self.previous_frame = frame.copy()
        return True

    def process_frame(self, frame_info: FrameInfo) -> Optional[Dict[str, Any]]:
        try:
            if self.is_valid_frame(frame_info.frame):
                self.last_valid_frame_time = time.time()
                
                start_time = time.time()
                result = self.board_processor.process_frame(frame_info.frame)
                processing_time = time.time() - start_time
                
                self.total_processing_time += processing_time
                self.frame_count += 1

                if result['state'] == BoardState.SHOWING:
                    sorted_board = sorted(result['board'], key=lambda card: (card['y'], card['x']))
                    card_values = [card['card'] for card in sorted_board]
                    logger.debug(f"Stable board: {card_values}")
                elif result['state'] == BoardState.NOT_SHOWING:
                    logger.debug("No board.")
                else:
                    logger.warning(f"Unexpected board state: {result['state']}")

                result = self.convert_enums_to_strings(result)
                result['frame_id'] = frame_info.frame_id
                result['frame_count'] = self.frame_count
                result['processing_time'] = processing_time

                logger.debug(f"Frame {result['frame_id']} | Processing time: {result['processing_time']:.3f}s")

                if not self.debug_mode and 'debug_info' in result:
                    del result['debug_info']

                if self.is_first_frame:
                    self.is_first_frame = False
                    logger.info("First frame processed successfully")

                return result
            else:
                logger.warning(f"Invalid frame detected. Time since last valid frame: {time.time() - self.last_valid_frame_time:.2f}s")
                return None
            
        except Exception as e:
            logger.error(f"Error in frame processing: {str(e)}", exc_info=True)
            raise

    def convert_enums_to_strings(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: self.convert_enums_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_enums_to_strings(item) for item in obj]
        return obj

    def check_for_quit(self):
        if self.config['platform'] == 'Windows':
            if self.kbhit():
                if self.getch() == b'q':
                    logger.info("Quit command received. Shutting down...")
                    return True
        else:
            try:
                c = sys.stdin.read(1)
                if c == 'q':
                    logger.info("Quit command received. Shutting down...")
                    return True
            except IOError:
                pass
        return False

    def run(self):

        try:
            while self.running and self.state == FrameProcessorState.RUNNING:
                frame_info = self.capture_frame()
                
                current_time = time.time()
                threshold = self.first_frame_threshold if self.is_first_frame else self.subsequent_frame_threshold
                # if self.frame_count > 1 and (current_time - self.last_valid_frame_time > threshold):
                #     logger.error(f"No valid frames received for {threshold} seconds. Shutting down.")
                #     self.state = FrameProcessorState.FAILED
                #    break

                if frame_info:
                    if frame_info.frame_id > self.last_processed_id:
                        result = self.process_frame(frame_info)
                        if result:
                            self.websocket_server.send_message(result)
                            self.last_processed_id = frame_info.frame_id
                    else:
                        logger.info(f"Skipping old frame {frame_info.frame_id}")
                else:
                    logger.warning("No frame captured")
                
                if self.check_for_quit():
                    break

                # Maintain target frame rate
                time_to_next_frame = self.frame_interval - (time.time() % self.frame_interval)
                if time_to_next_frame > 0:
                    time.sleep(time_to_next_frame)

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}", exc_info=True)
            self.state = FrameProcessorState.FAILED
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("Initiating shutdown...")
        self.running = False
        if self.config['capture']['mode'] == 'obs_websocket':
            self.obs_client.disconnect()
        elif hasattr(self.config['capture'], 'camera_manager'):
            self.config['capture']['camera_manager'].release_camera()
        
        if self.config['platform'] != 'Windows':
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_settings)
        
        end_time = time.time()
        total_run_time = end_time - self.start_time
        logger.info(f"Total run time: {total_run_time:.2f}s")
        logger.info(f"Total frames processed: {self.frame_count}")
        if self.frame_count > 0:
            avg_processing_time = self.total_processing_time / self.frame_count
            logger.info(f"Average processing time: {avg_processing_time:.4f}s")
            logger.info(f"Effective FPS: {self.frame_count/total_run_time:.2f}")
        logger.info("Shutdown complete")