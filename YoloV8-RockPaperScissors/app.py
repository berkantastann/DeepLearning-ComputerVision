from ultralytics import YOLO
import cv2 as cv
model = YOLO("/Users/berkantastan/Desktop/DeepLearning-ComputerVision/runs/detect/custom_model/weights/best.pt")

cap = cv. VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    result = model.track(frame, conf=0.5, persist=True, device='mps')
    annotated_frame = result[0].plot()
    cv.imshow("YOLOv8 Detection", annotated_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()