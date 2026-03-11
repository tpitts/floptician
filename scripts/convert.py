from ultralytics import YOLO
from _common import models_path

model = YOLO(str(models_path("best38-m.pt")))  # Load the YOLOv8 model
model.export(format="coreml", imgsz=640)  # Export to CoreML format
