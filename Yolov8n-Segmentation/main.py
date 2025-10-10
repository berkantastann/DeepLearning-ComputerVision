from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.pt")  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.5)
    result = results[0]
    masks = result.masks

    annotated_frame = frame.copy()

    if masks is not None:
        for i, mask in enumerate(masks.data):
            mask = mask.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask > 0] = color

            annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

    cv2.imshow("YOLOv8 Segmentation", annotated_frame)

    if cv2.waitKey(7) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()