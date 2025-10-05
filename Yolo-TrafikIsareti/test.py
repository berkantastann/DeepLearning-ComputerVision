from ultralytics import YOLO
import cv2 as cv

model = YOLO("/Users/berkantastan/Desktop/DeepLearning-ComputerVision/Yolo-TrafikIsareti/runs/detect/traffic_sign_model/weights/best.pt")

image_path = "test.jpg"
image = cv.imread(image_path)

results = model(image)[0]
print(results)

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0]
    cls = int(box.cls[0])
    label = f"{model.names[cls]} {conf:.2f}"
    
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(image, label , (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv.imshow("Image", image)
    cv.waitKey(0)
    
    cv.imwrite("output.jpg", image)
cv.destroyAllWindows()   