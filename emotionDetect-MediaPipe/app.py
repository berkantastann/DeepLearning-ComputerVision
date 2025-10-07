import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def detect_emotion(image):
    """
    MediaPipe yüz landmarklarına göre kural tabanlı duygu analizi yapar.
    """
    # Görüntüyü RGB'ye çevir
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return "No Face Detected"

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    # --- Önemli Noktalar (FaceMesh indexlerine göre) ---
    # Göz, kaş, ağız landmark’ları
    left_eye_top = (landmarks[159].x * w, landmarks[159].y * h)
    left_eye_bottom = (landmarks[145].x * w, landmarks[145].y * h)

    right_eye_top = (landmarks[386].x * w, landmarks[386].y * h)
    right_eye_bottom = (landmarks[374].x * w, landmarks[374].y * h)

    mouth_top = (landmarks[13].x * w, landmarks[13].y * h)
    mouth_bottom = (landmarks[14].x * w, landmarks[14].y * h)

    left_eyebrow = (landmarks[70].x * w, landmarks[70].y * h)
    left_eye_center = (landmarks[33].x * w, landmarks[33].y * h)

    # --- Mesafe Hesapları ---
    eye_open_left = euclidean_distance(left_eye_top, left_eye_bottom)
    eye_open_right = euclidean_distance(right_eye_top, right_eye_bottom)
    eye_open = (eye_open_left + eye_open_right) / 2

    mouth_open = euclidean_distance(mouth_top, mouth_bottom)
    brow_height = euclidean_distance(left_eyebrow, left_eye_center)

    # --- Normalize için yüz yüksekliği (yaklaşık) ---
    face_height = euclidean_distance(
        (landmarks[10].x * w, landmarks[10].y * h),   # Alın
        (landmarks[152].x * w, landmarks[152].y * h)  # Çene
    )

    eye_ratio = eye_open / face_height
    mouth_ratio = mouth_open / face_height
    brow_ratio = brow_height / face_height

    # --- Basit Kurallar ---
    if mouth_ratio > 0.08 and eye_ratio > 0.05:
        emotion = "Surprised 😮"
    elif mouth_ratio > 0.06:
        emotion = "Happy 😀"
    elif brow_ratio < 0.08 and mouth_ratio < 0.04:
        emotion = "Sad 😔"
    else:
        emotion = "Neutral 😐"

    return emotion

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    emotion = detect_emotion(frame)
    cv2.putText(frame, emotion, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()