import platform
import cv2
import subprocess

# Importing necessary libraries for macOS
if platform.system() == 'Darwin':
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

# For Windows, we use pygrabber to detect cameras. This is only imported on Windows systems.
if platform.system() == 'Windows':
    from pygrabber.dshow_graph import FilterGraph

class CameraManager:
    """
    A cross-platform camera management class.
    
    This class provides functionality to:
    1. Detect available cameras on the system.
    2. Get camera names (uses pygrabber on Windows for more accurate names, and AVFoundation on macOS).
    3. Open a selected camera.
    4. Capture and display video frames.

    The class is designed to work across different platforms (Windows, Linux, macOS),
    but some features (like detailed camera names) are platform-specific.
    """

    def __init__(self):
        self.cameras = []
        self.current_camera = None

    def get_available_cameras(self):
        """
        Detects and returns a list of available cameras.
        
        Returns:
        List of dicts, each containing 'camera_index' and 'camera_name'.
        """
        if platform.system() == 'Windows':
            cameras = self._get_windows_cameras()
        elif platform.system() == 'Darwin':  # macOS specific method
            cameras = self._get_macos_cameras()
        else:
            cameras = self._get_generic_cameras()

        # Sort cameras by their names to ensure consistent ordering
        cameras.sort(key=lambda cam: cam['camera_name'])
        return cameras

    def _get_macos_cameras(self):
        """
        macOS-specific method to get camera information using AVFoundation.
        """
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
        return [{'camera_index': i, 'camera_name': device.localizedName()} for i, device in enumerate(devices)]

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

    def open_camera(self, camera_index):
        """
        Opens the selected camera.
        
        Args:
        camera_index (int): The index of the camera to open.

        Returns:
        bool: True if camera opened successfully, False otherwise.
        """
        if platform.system() == 'Windows':
            self.current_camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            self.current_camera = cv2.VideoCapture(camera_index)
        
        return self.current_camera.isOpened()

    def set_resolution(self, width, height):
        """
        Sets the resolution of the current camera.

        Args:
        width (int): Desired width of the camera frame.
        height (int): Desired height of the camera frame.
        """
        if self.current_camera:
            self.current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        """
        Captures a frame from the current camera.

        Returns:
        tuple: (success (bool), frame (numpy array))
        """
        if self.current_camera:
            return self.current_camera.read()
        return False, None

    def release_camera(self):
        """
        Releases the current camera.
        """
        if self.current_camera:
            self.current_camera.release()
            self.current_camera = None

    def __del__(self):
        """
        Destructor to ensure camera is released when the object is deleted.
        """
        self.release_camera()
