import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
print("Loading YOLO model...")
model = YOLO("..\\models\\best.pt")
print("YOLO model loaded successfully.")

def detect_and_create_overlay(frame):
    results = model(frame)
    overlay = frame.copy()
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            conf = float(score)
            label_name = model.names[int(label)]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{label_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return overlay

def list_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(index)
        cap.release()
        index += 1
    return cameras

def main():
    # List available cameras
    available_cameras = list_cameras()
    
    print("Available cameras:")
    for i, cam in enumerate(available_cameras):
        print(f"{i}: Camera {cam}")
    
    # Prompt user to select a camera
    selected_camera = int(input("Select a camera by number: "))
    if selected_camera not in range(len(available_cameras)):
        print("Invalid camera selection.")
        return

    cap = cv2.VideoCapture(available_cameras[selected_camera])

    if not cap.isOpened():
        print("Error: Could not open selected webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Webcam opened: {width}x{height} at {fps} FPS")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            overlay = detect_and_create_overlay(frame)
            
            # Display the frame with overlay
            cv2.imshow('Webcam Feed with Overlay', overlay)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit command received.")
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Overlay creation ended.")

if __name__ == "__main__":
    main()