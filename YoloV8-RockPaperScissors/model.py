from ultralytics import YOLO

model = YOLO("yolov8n.pt")

data ="data/data.yaml"

model.train(
    data=data,
    epochs=2,
    imgsz=640,
    batch=16,
    verbose=True,
    name ="custom_model",
)