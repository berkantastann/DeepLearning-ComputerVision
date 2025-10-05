from ultralytics import YOLO
import cv2 as cv


model = YOLO("yolov8n.pt")

video_path = "/Users/berkantastan/Desktop/DeepLearning-ComputerVision/Yolo-ObjectTracking/car/IMG_5269.MOV"

cap = cv.VideoCapture(video_path)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

out = cv.VideoWriter("output.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, conf=0.3, iou=0.5, device="mps", tracker="bytetrack.yaml")
    
    annotated_frame = results[0].plot()
    
    out.write(annotated_frame)
    cv.imshow("YOLOv8 Tracking", annotated_frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
out.release()
cv.destroyAllWindows()
        
