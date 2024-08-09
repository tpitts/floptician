import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict
import cv2
import numpy as np
from enum import Enum  # Add this import
from aioconsole import ainput  # Add this import
from app.board_processor import BoardProcessor, BoardState
from app.obs_client import OBSClient
from app.camera_manager import CameraManager
from app.custom_websocket_server import WebSocketServer

logger = logging.getLogger(__name__)

@dataclass
class FrameInfo:
    frame_id: int
    frame: Any
    capture_time: float

class FrameProcessor:
    """
    Manages the capture, processing, and distribution of video frames.
    Handles concurrent operations using asyncio for improved performance.
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
        
        # Set up concurrency control
        self.capture_semaphore = asyncio.Semaphore(3)  # Limit concurrent capture operations
        self.processing_semaphore = asyncio.Semaphore(2)  # Limit concurrent processing operations
        
        # Initialize queues for inter-task communication
        self.frame_queue = asyncio.Queue()
        self.websocket_queue = asyncio.Queue()
        
        # Set up runtime control
        self.running = asyncio.Event()
        self.running.set()
        
        # Initialize performance tracking
        self.total_time = 0
        self.frame_count = 0
        self.start_time = time.time()

    async def capture_frame(self) -> FrameInfo:
        """
        Capture a frame from the configured source (OBS or direct camera).
        Uses a semaphore to limit concurrent capture operations.
        """
        async with self.capture_semaphore:
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

    async def process_frame(self, frame_info: FrameInfo) -> Dict[str, Any]:
        """
        Process a captured frame using the board processor.
        Uses a semaphore to limit concurrent processing operations.
        """
        async with self.processing_semaphore:
            try:
                start_time = time.time()
                result = self.board_processor.process_frame(frame_info.frame)
                processing_time = time.time() - start_time
                
                # Update performance metrics
                self.total_time += processing_time
                self.frame_count += 1

                # Log results based on board state
                if result['state'] == BoardState.SHOWING:
                    logger.info(f"Stable board detected: {result['board']}")
                elif result['state'] == BoardState.NOT_SHOWING:
                    logger.info("No board detected")
                else:
                    logger.warning(f"Unexpected board state: {result['state']}")

                # Prepare result for transmission
                result = self.convert_enums_to_strings(result)
                result['frame_id'] = frame_info.frame_id
                result['frame_count'] = self.frame_count
                result['processing_time'] = processing_time

                logger.debug(f"Frame {self.frame_count} processed: {result}")

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

    async def frame_producer(self):
        """
        Continuously capture frames and put them into the frame queue.
        Maintains the target frame rate using asyncio.sleep().
        """
        while self.running.is_set():
            try:
                frame_info = await self.capture_frame()
                if frame_info and frame_info.frame is not None:
                    await self.frame_queue.put(frame_info)
                await asyncio.sleep(self.frame_interval)
            except Exception as e:
                logger.error(f"Error in frame producer: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Wait before retrying to avoid rapid error loops

    async def frame_consumer(self):
        """
        Consume frames from the queue, process them, and queue results for websocket transmission.
        Drops old frames to maintain real-time processing.
        """
        while self.running.is_set():
            try:
                frame_info = await self.frame_queue.get()
                if frame_info.frame_id > self.last_processed_id:
                    result = await self.process_frame(frame_info)
                    if result:
                        await self.websocket_queue.put(result)
                        self.last_processed_id = frame_info.frame_id
                else:
                    logger.info(f"Dropping old frame {frame_info.frame_id}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in frame consumer: {str(e)}", exc_info=True)
            finally:
                self.frame_queue.task_done()

    async def websocket_sender(self):
        """
        Send processed results to connected websocket clients.
        """
        while self.running.is_set():
            try:
                result = await self.websocket_queue.get()
                self.websocket_server.send_message(result)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in websocket sender: {str(e)}", exc_info=True)
            finally:
                self.websocket_queue.task_done()

    async def check_for_quit(self):
        """
        Check for user input to quit the application.
        Uses a short timeout to allow frequent checks of the running flag.
        """
        while self.running.is_set():
            try:
                user_input = await asyncio.wait_for(ainput(), timeout=0.1)
                if user_input.lower() == 'q':
                    logger.info("Quit command received. Shutting down...")
                    self.running.clear()
                    break
            except asyncio.TimeoutError:
                pass  # This allows checking self.running.is_set() frequently

    async def shutdown(self):
        """
        Gracefully shut down all running tasks and release resources.
        """
        logger.info("Initiating shutdown...")
        self.running.clear()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.config['capture']['mode'] == 'obs_websocket':
            self.obs_client.disconnect()
        elif hasattr(self.config['capture'], 'camera_manager'):
            self.config['capture']['camera_manager'].release_camera()
        logger.info("Shutdown complete")

    async def run(self):
        """
        Main run method to start all asynchronous tasks.
        Handles exceptions and performs cleanup on exit.
        """
        tasks = [
            asyncio.create_task(self.frame_producer()),
            asyncio.create_task(self.frame_consumer()),
            asyncio.create_task(self.websocket_sender()),
            asyncio.create_task(self.check_for_quit())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled. Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}", exc_info=True)
        finally:
            await self.shutdown()
            end_time = time.time()
            total_run_time = end_time - self.start_time
            logger.info(f"Total run time: {total_run_time:.2f}s")
            logger.info(f"Total frames processed: {self.frame_count}")
            if self.frame_count > 0:
                logger.info(f"Average processing time: {self.total_time/self.frame_count:.4f}s")
                logger.info(f"Effective FPS: {self.frame_count/total_run_time:.2f}")