from ultralytics import YOLO
import cv2 as cv

# Modeli yükle
model = YOLO("yolov8n.pt")

# Video yolu
video_path = "/Users/berkantastan/Desktop/DeepLearning-ComputerVision/Yolo-ObjectTracking/car/IMG_5270.MOV"
cap = cv.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ----------- PREDICT: sadece tespit -----------
    results_predict = model.predict(frame, conf=0.3, device="mps")
    frame_predict = results_predict[0].plot()  # sadece bounding box çizer

    # ----------- TRACK: tespit + takip -----------
    results_track = model.track(frame, persist=True, conf=0.3, device="mps",tracker="bytetrack.yaml")
    frame_track = results_track[0].plot()  # hem box hem ID çizer

    # ---------- Görselleri yan yana birleştir ----------
    combined = cv.hconcat([frame_predict, frame_track])

    # Başlık ekleyelim
    cv.putText(combined, "Predict (sol)  |  Track (sag)", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv.imshow("Predict vs Track", combined)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()