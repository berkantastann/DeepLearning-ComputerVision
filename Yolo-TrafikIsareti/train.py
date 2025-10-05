from ultralytics import YOLO

# Modeli seç
model = YOLO("yolov8n.pt")  # veya yolov8s.pt

# Veri seti yolunu ayarla
data = "traffic-sign-detection.yolov8/data.yaml"

# Modeli eğit
model.train(
    data=data,
    epochs=2,
    imgsz=640,
    batch=16,
    verbose=True,
    name ="traffic_sign_model",
)