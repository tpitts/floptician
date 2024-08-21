from ultralytics import YOLO

model = YOLO("./models/best.pt")  # Load the YOLOv8 model
model.export(format="coreml")  # Export to CoreML format