import os
import cv2
import numpy as np

# YOLO formatına dönüştürme fonksiyonu
def mask_to_yolo(mask_path, save_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape[:2]

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if len(contour) < 3: 
            continue
        
        coords = contour.squeeze().astype(float)
        coords[:, 0] /= w  # x
        coords[:, 1] /= h  # y

        coords_str = " ".join(map(str, coords.flatten()))
        line = f"{class_id} {coords_str}"
        lines.append(line)
    
    # TXT olarak kaydet
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


# Train ve Valid klasörleri
for split in ["train", "valid"]:
    mask_dir = f"data/{split}/masks"
    label_dir = f"data/{split}/labels"
    os.makedirs(label_dir, exist_ok=True)

    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith((".png", ".jpg", ".jpeg")):
            mask_path = os.path.join(mask_dir, mask_file)
            label_path = os.path.join(label_dir, mask_file.rsplit(".", 1)[0] + ".txt")
            
            mask_to_yolo(mask_path, label_path, class_id=0)

print("✅ Tüm maskeler YOLO segmentation formatına dönüştürüldü!")