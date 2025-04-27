import platform
import cv2
import subprocess
import time
import threading
import numpy as np
from queue import Queue, Empty

# For Windows, we use pygrabber to detect cameras. This is only imported on Windows systems.
if platform.system() == 'Windows':
    from pygrabber.dshow_graph import FilterGraph


class CameraManager:
    """
    A cross-platform camera management class.
    
    This class provides functionality to:
    1. Detect available cameras on the system.
    2. Get camera names (uses pygrabber on Windows for more accurate names, and platform-specific methods for macOS).
    3. Open a selected camera with robust handling on macOS.
    4. Capture and display video frames.

    The class is designed to work across different platforms (Windows, Linux, macOS),
    with robust camera handling that maintains connection stability when devices are
    added or removed during operation.
    """

    def __init__(self):
        self.cameras = []
        self.current_camera = None
        self.platform = platform.system()
        self.current_camera_index = None
        self.current_device_id = None
        
        # Store mappings between indices and device IDs for macOS
        self.device_id_to_index = {}
        self.index_to_device_id = {}
        
        # Initialize the camera list
        self.refresh_camera_list()

    def refresh_camera_list(self):
        """
        Refreshes the list of available cameras.
        
        Call this method when you suspect camera configurations have changed.
        """
        if self.platform == 'Windows':
            self.cameras = self._get_windows_cameras()
        elif self.platform == 'Darwin':  # macOS specific method
            self.cameras = self._get_macos_cameras()
        else:
            self.cameras = self._get_generic_cameras()

        # Sort cameras by their names to ensure consistent ordering
        self.cameras.sort(key=lambda cam: cam['camera_name'])
        
        # Update mappings for macOS
        if self.platform == 'Darwin':
            self.device_id_to_index = {}
            self.index_to_device_id = {}
            for camera in self.cameras:
                if 'device_id' in camera:
                    self.device_id_to_index[camera['device_id']] = camera['camera_index']
                    self.index_to_device_id[camera['camera_index']] = camera['device_id']
        
        return self.cameras
        
    def get_available_cameras(self):
        """
        Returns the list of available cameras.
        
        If you want to refresh the list first, call refresh_camera_list().
        
        Returns:
        List of dicts containing camera information.
        """
        return self.cameras

    def _get_macos_cameras(self):
        """
        macOS-specific method to get camera information.
        
        Returns a list of dictionaries with camera information including
        persistent device IDs to maintain connections even if cameras
        are added/removed.
        """
        cameras = []
        try:
            # Only import these for macOS
            import objc
            import AVFoundation
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
            
            devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
            
            for i, device in enumerate(devices):
                cameras.append({
                    'camera_index': i,  # Still maintain an index for compatibility
                    'camera_name': device.localizedName(),
                    'device_id': str(device.uniqueID())  # Store the persistent unique ID
                })
        except Exception as e:
            print(f"Error getting macOS cameras: {e}")
            # Fallback to basic OpenCV detection
            cameras = self._get_generic_cameras()
        
        return cameras

    def _get_windows_cameras(self):
        """
        Windows-specific method to get camera information using pygrabber.
        """
        graph = FilterGraph()
        devices = graph.get_input_devices()
        return [{'camera_index': i, 'camera_name': name} for i, name in enumerate(devices)]

    def _get_generic_cameras(self):
        """
        Generic method to detect cameras using OpenCV, used for non-Windows and non-macOS platforms.
        """
        camera_indexes = self._get_camera_indexes()
        return self._add_camera_information(camera_indexes)

    def _get_camera_indexes(self):
        """
        Helper method to find available camera indexes using OpenCV.
        """
        index = 0
        camera_indexes = []
        max_cameras_to_check = 10
        while max_cameras_to_check > 0:
            capture = cv2.VideoCapture(index)
            if capture.read()[0]:
                camera_indexes.append(index)
                capture.release()
            index += 1
            max_cameras_to_check -= 1
        return camera_indexes

    def _add_camera_information(self, camera_indexes):
        """
        Helper method to add camera names to indexes.
        Uses system commands on Linux for more detailed names.
        """
        cameras = []
        for camera_index in camera_indexes:
            if platform.system() == 'Linux':
                try:
                    camera_name = subprocess.run(['cat', f'/sys/class/video4linux/video{camera_index}/name'],
                                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout.decode('utf-8').strip()
                except subprocess.CalledProcessError:
                    camera_name = f'Camera {camera_index}'
            else:
                camera_name = f'Camera {camera_index}'
            cameras.append({'camera_index': camera_index, 'camera_name': camera_name})
        return cameras
    
    def find_camera_by_id(self, device_id):
        """
        Find a camera by its unique device ID (macOS only).
        
        Args:
        device_id (str): The unique ID of the camera device.
        
        Returns:
        dict: Camera information if found, None otherwise.
        """
        if self.platform != 'Darwin':
            return None
            
        for camera in self.cameras:
            if camera.get('device_id') == device_id:
                return camera
        return None
        
    def open_camera(self, camera_identifier):
        """
        Opens the selected camera.
        
        Args:
        camera_identifier: 
            - On macOS: can be either a camera_index or a device_id string
            - On other platforms: must be a camera_index (int)

        Returns:
        bool: True if camera opened successfully, False otherwise.
        """
        self.release_camera()  # Make sure to release any previously opened camera
        
        if self.platform == 'Darwin':
            # For macOS, handle the device ID tracking
            if isinstance(camera_identifier, str):
                # Using device_id string
                self.current_device_id = camera_identifier
                
                # Find the corresponding index
                camera_index = self.device_id_to_index.get(camera_identifier)
                if camera_index is None:
                    # If we don't have a mapping, refresh and check again
                    self.refresh_camera_list()
                    camera_index = self.device_id_to_index.get(camera_identifier)
                    if camera_index is None:
                        print(f"Could not find camera with device ID: {camera_identifier}")
                        return False
            else:
                # Using camera_index
                camera_index = camera_identifier
                self.current_camera_index = camera_index
                
                # Store the device_id for this index if available
                self.current_device_id = self.index_to_device_id.get(camera_index)
            
            # Use OpenCV to open the camera by index
            self.current_camera = cv2.VideoCapture(camera_index)
            self.current_camera_index = camera_index
            return self.current_camera.isOpened()
            
        elif self.platform == 'Windows':
            # For Windows, use DirectShow
            self.current_camera = cv2.VideoCapture(camera_identifier, cv2.CAP_DSHOW)
            self.current_camera_index = camera_identifier
            return self.current_camera.isOpened()
        else:
            # For other platforms
            self.current_camera = cv2.VideoCapture(camera_identifier)
            self.current_camera_index = camera_identifier
            return self.current_camera.isOpened()

    def set_resolution(self, width, height):
        """
        Sets the resolution of the current camera.

        Args:
        width (int): Desired width of the camera frame.
        height (int): Desired height of the camera frame.
        """
        if self.current_camera:
            # Use OpenCV to set resolution
            self.current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify the resolution was set
            actual_width = self.current_camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.current_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if abs(actual_width - width) > 10 or abs(actual_height - height) > 10:
                print(f"Warning: Requested resolution {width}x{height} but got {actual_width}x{actual_height}")
                
            return True
        return False

    def get_frame(self):
        """
        Captures a frame from the current camera.
        Includes recovery mechanisms for macOS.

        Returns:
        tuple: (success (bool), frame (numpy array))
        """
        if not self.current_camera:
            return False, None
            
        success, frame = self.current_camera.read()
        
        # Attempt recovery if frame capture fails on macOS
        if not success and self.platform == 'Darwin' and self.current_device_id:
            print("Camera read failed, attempting recovery...")
            
            # Release and refresh
            self.release_camera()
            self.refresh_camera_list()
            
            # Try to reopen with the same device ID
            if self.current_device_id in self.device_id_to_index:
                camera_index = self.device_id_to_index[self.current_device_id]
                self.current_camera = cv2.VideoCapture(camera_index)
                self.current_camera_index = camera_index
                
                # Try again
                return self.current_camera.read()
        
        return success, frame

    def release_camera(self):
        """
        Releases the current camera.
        """
        if self.current_camera:
            self.current_camera.release()
            self.current_camera = None
        
        # Don't reset these so we can recover if needed
        # self.current_camera_index = None
        # self.current_device_id = None

    def check_camera_status(self):
        """
        Checks if the current camera is still valid and working.
        Useful to call periodically to detect camera disconnections.
        
        Returns:
        bool: True if camera is working, False otherwise
        """
        if not self.current_camera:
            return False
            
        # Try to grab a frame without decoding it (faster than read())
        status = self.current_camera.grab()
        
        # If grab fails on macOS and we have a device ID, try to recover
        if not status and self.platform == 'Darwin' and self.current_device_id:
            self.refresh_camera_list()
            
            # Check if our device still exists
            if self.current_device_id in self.device_id_to_index:
                # Device exists, try to reopen it
                self.release_camera()
                
                camera_index = self.device_id_to_index[self.current_device_id]
                self.current_camera = cv2.VideoCapture(camera_index)
                self.current_camera_index = camera_index
                
                return self.current_camera.isOpened()
                
        return status
        
    def start_monitoring(self, interval=2.0):
        """
        Start a background thread that monitors for camera changes.
        
        Args:
        interval (float): How often to check for changes (in seconds)
        """
        if self.platform != 'Darwin':
            return  # Only needed for macOS
            
        def monitor_thread():
            while True:
                old_cameras = self.cameras.copy()
                new_cameras = self._get_macos_cameras()
                
                # Check if the camera list has changed
                if len(old_cameras) != len(new_cameras):
                    self._handle_camera_change()
                else:
                    # Check if any device IDs have changed position
                    old_ids = [c.get('device_id') for c in old_cameras]
                    new_ids = [c.get('device_id') for c in new_cameras]
                    
                    if old_ids != new_ids:
                        self._handle_camera_change()
                        
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_thread, daemon=True)
        self.monitor_thread.start()
        
    def _handle_camera_change(self):
        """Handle camera configuration changes."""
        print("Camera configuration change detected")
        
        # Store current device ID
        current_id = self.current_device_id
        
        # Refresh camera list
        self.refresh_camera_list()
        
        # If we're currently using a camera and have its ID
        if self.current_camera and current_id:
            # Check if our device still exists
            if current_id in self.device_id_to_index:
                # Device still exists, check if index changed
                new_index = self.device_id_to_index[current_id]
                
                # If the index changed, we need to reopen
                if self.current_camera_index != new_index:
                    self.release_camera()
                    self.open_camera(current_id)
            else:
                # Our device is gone, release the camera
                self.release_camera()

    def __del__(self):
        """
        Destructor to ensure camera is released when the object is deleted.
        """
        self.release_camera()
