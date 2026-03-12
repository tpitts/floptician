from __future__ import annotations

import logging
import os
import platform
import sys
import time

import cv2
import numpy as np

from floptician.models import AppConfig, BoardResult, BoardState, CaptureMode, FrameInfo, FrameProcessorState
from floptician.protocols import MessageBroadcaster, OBSProtocol

logger = logging.getLogger(__name__)

# Move Unix-specific imports to the top
if platform.system() != "Windows":
    import fcntl
    import termios


class FrameProcessor:
    def __init__(
        self, config: AppConfig, obs_client: OBSProtocol, board_processor, websocket_server: MessageBroadcaster
    ):
        self.config = config
        self.obs_client = obs_client
        self.board_processor = board_processor
        self.websocket_server = websocket_server

        self.target_fps = config.capture.fps
        self.frame_interval = 1 / self.target_fps

        self.last_frame_id = 0
        self.last_processed_id = 0

        self.total_processing_time = 0
        self.frame_count = 0

        self.state = FrameProcessorState.RUNNING
        self.running = True

        self.start_time = time.time()
        self.last_valid_frame_time = self.start_time

        self.first_frame_threshold = 8.0
        self.subsequent_frame_threshold = 4.0
        self.is_first_frame = True

        self.debug_mode = config.debug

        self.previous_frame = None

        if config.platform == "Windows":
            import msvcrt

            self.kbhit = msvcrt.kbhit
            self.getch = msvcrt.getch
        else:
            logger.debug("Loading Unix-specific input handling modules...")
            import termios

            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            self.setup_unix_input()

    def setup_unix_input(self):
        new_settings = termios.tcgetattr(self.fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSANOW, new_settings)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, fcntl.fcntl(self.fd, fcntl.F_GETFL) | os.O_NONBLOCK)

    def capture_frame(self) -> FrameInfo | None:
        try:
            if self.config.capture.mode == CaptureMode.OBS_WEBSOCKET:
                image_data = self.obs_client.capture_frame(self.config.capture.selected_webcam)
                if image_data:
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    frame = None
            else:
                success, frame = self.config.capture.camera_manager.get_frame()
                if not success:
                    frame = None

            self.last_frame_id += 1
            return FrameInfo(self.last_frame_id, frame, time.time()) if frame is not None else None
        except Exception as e:
            logger.error(f"Error in frame capture: {e!s}", exc_info=True)
            return None

    def is_valid_frame(self, frame: np.ndarray) -> bool:
        if frame is None:
            logger.warning("No frame captured")
            return False

        total_pixels = frame.shape[0] * frame.shape[1]
        black_pixels = np.sum(frame == 0) / 3
        white_pixels = np.sum(frame == 255) / 3

        if black_pixels / total_pixels > 0.5:
            logger.warning("Frame is over 50% black")
            return False

        if white_pixels / total_pixels > 0.5:
            logger.warning("Frame is over 50% white")
            return False

        if self.previous_frame is not None and np.array_equal(frame, self.previous_frame):
            logger.warning("Frozen frame detected")
            return False

        self.previous_frame = frame.copy()
        return True

    def process_frame(self, frame_info: FrameInfo) -> dict | None:
        try:
            if self.is_valid_frame(frame_info.frame):
                self.last_valid_frame_time = time.time()

                start_time = time.time()
                result: BoardResult = self.board_processor.process_frame(frame_info.frame)
                processing_time = time.time() - start_time

                self.total_processing_time += processing_time
                self.frame_count += 1

                if result.state == BoardState.SHOWING:
                    sorted_board = sorted(result.board, key=lambda card: (card.y, card.x))
                    card_values = [card.card for card in sorted_board]
                    logger.debug(f"Stable board: {card_values}")
                elif result.state == BoardState.NOT_SHOWING:
                    logger.debug("No board.")
                else:
                    logger.warning(f"Unexpected board state: {result.state}")

                result_dict = result.to_dict()
                result_dict["frame_id"] = frame_info.frame_id
                result_dict["frame_count"] = self.frame_count
                result_dict["processing_time"] = processing_time

                logger.debug(f"Frame {result_dict['frame_id']} | Processing time: {processing_time:.3f}s")

                if not self.debug_mode and "debug_info" in result_dict:
                    del result_dict["debug_info"]

                if self.is_first_frame:
                    self.is_first_frame = False
                    logger.info("First frame processed successfully")

                return result_dict
            else:
                gap = time.time() - self.last_valid_frame_time
                logger.warning(f"Invalid frame detected. Time since last valid frame: {gap:.2f}s")
                return None

        except Exception as e:
            logger.error(f"Error in frame processing: {e!s}", exc_info=True)
            raise

    def check_for_quit(self):
        if self.config.platform == "Windows":
            if self.kbhit() and self.getch() == b"q":
                logger.info("Quit command received. Shutting down...")
                return True
        else:
            try:
                c = sys.stdin.read(1)
                if c == "q":
                    logger.info("Quit command received. Shutting down...")
                    return True
            except OSError:
                pass
        return False

    def run(self):
        try:
            while self.running and self.state == FrameProcessorState.RUNNING:
                frame_info = self.capture_frame()

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

                time_to_next_frame = self.frame_interval - (time.time() % self.frame_interval)
                if time_to_next_frame > 0:
                    time.sleep(time_to_next_frame)

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e!s}", exc_info=True)
            self.state = FrameProcessorState.FAILED
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("Initiating shutdown...")
        self.running = False
        if self.config.capture.mode == CaptureMode.OBS_WEBSOCKET:
            self.obs_client.disconnect()
        elif hasattr(self.config.capture, "camera_manager"):
            self.config.capture.camera_manager.release_camera()

        if self.config.platform != "Windows":
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_settings)

        end_time = time.time()
        total_run_time = end_time - self.start_time
        logger.info(f"Total run time: {total_run_time:.2f}s")
        logger.info(f"Total frames processed: {self.frame_count}")
        if self.frame_count > 0:
            avg_processing_time = self.total_processing_time / self.frame_count
            logger.info(f"Average processing time: {avg_processing_time:.4f}s")
            logger.info(f"Effective FPS: {self.frame_count / total_run_time:.2f}")
        logger.info("Shutdown complete")
