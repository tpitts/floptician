import platform
import subprocess
import cv2

# Comment: We aim for cross-platform compatibility, but use pygrabber on Windows for easy camera name retrieval.
if platform.system() == 'Windows':
    from pygrabber.dshow_graph import FilterGraph

class Camera:
    def __init__(self):
        self.cameras = []

    def get_camera_info(self) -> list:
        self.cameras = []
        if platform.system() == 'Windows':
            self.cameras = self.get_windows_cameras()
        else:
            camera_indexes = self.get_camera_indexes()
            if camera_indexes:
                self.cameras = self.add_camera_information(camera_indexes)
        return self.cameras

    def get_windows_cameras(self):
        graph = FilterGraph()
        devices = graph.get_input_devices()
        return [{'camera_index': i, 'camera_name': name} for i, name in enumerate(devices)]

    def get_camera_indexes(self):
        index = 0
        camera_indexes = []
        max_numbers_of_cameras_to_check = 10
        while max_numbers_of_cameras_to_check > 0:
            capture = cv2.VideoCapture(index)
            if capture.read()[0]:
                camera_indexes.append(index)
                capture.release()
            index += 1
            max_numbers_of_cameras_to_check -= 1
        return camera_indexes

    def add_camera_information(self, camera_indexes: list) -> list:
        platform_name = platform.system()
        cameras = []

        if platform_name == 'Linux':
            for camera_index in camera_indexes:
                try:
                    camera_name = subprocess.run(['cat', f'/sys/class/video4linux/video{camera_index}/name'],
                                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout.decode('utf-8')
                    camera_name = camera_name.replace('\n', '')
                except subprocess.CalledProcessError:
                    camera_name = f'Camera {camera_index}'
                cameras.append({'camera_index': camera_index, 'camera_name': camera_name})
        else:  # For macOS or other platforms
            for camera_index in camera_indexes:
                cameras.append({'camera_index': camera_index, 'camera_name': f'Camera {camera_index}'})

        return cameras

def main():
    camera = Camera()
    available_cameras = camera.get_camera_info()

    if not available_cameras:
        print("No cameras found.")
        return

    print("Available cameras:")
    for i, cam in enumerate(available_cameras):
        print(f"{i}: {cam['camera_name']} (Index: {cam['camera_index']})")

    while True:
        try:
            selection = int(input("Select a camera by entering its number: "))
            if 0 <= selection < len(available_cameras):
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    selected_camera = available_cameras[selection]['camera_index']
    print(f"Attempting to open camera {available_cameras[selection]['camera_name']}...")
    
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(selected_camera)

    if not cap.isOpened():
        print(f"Failed to open camera {available_cameras[selection]['camera_name']}.")
        return

    # Set resolution to 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get the actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"\nActual resolution: {actual_width}x{actual_height}")

    print("Camera opened successfully. Attempting to read frames...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame. Frame count: {frame_count}")
            break

        frame_count += 1
        cv2.putText(frame, f"Frame {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Webcam', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

if __name__ == "__main__":
    main()