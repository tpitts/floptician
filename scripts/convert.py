from ultralytics import YOLO

model = YOLO("./models/best.pt")  # Load the YOLOv8 model
model.export(format="coreml", imgsz=640)  # Export to CoreML format