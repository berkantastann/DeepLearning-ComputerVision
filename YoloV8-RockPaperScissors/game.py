from  ultralytics import YOLO
import time 
import cv2 as cv
import random

cap = cv.VideoCapture(0)
model = YOLO("best.onnx")    

skor = {"oyuncu": 0, "bilgisayar": 0}
secenekler = ["paper", "rock", "scissors"]
son_oyun_zamani = 0
bekleme_suresi = 3

kazanan_durumlar = [("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")]

def kim_kazandi(oyuncu_secimi, bilgisayar_secimi):
    if (oyuncu_secimi == bilgisayar_secimi):
        return "berabere"
    elif (oyuncu_secimi, bilgisayar_secimi) in kazanan_durumlar:
        skor["oyuncu"] += 1
        return "kazandın"
    else:
        skor["bilgisayar"] += 1
        return "bilgisayar kazandı"
    

bilgisayar_secimi = None
Sonuc = None


while True:
    ret,frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame)[0]
    oyuncu_secimi = None
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        if conf > 0.6:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            oyuncu_secimi = secenekler[int(cls)]
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, oyuncu_secimi, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            break
    if oyuncu_secimi and (time.time() - son_oyun_zamani) > bekleme_suresi:
        bilgisayar_secimi = random.choice(secenekler)
        Sonuc = kim_kazandi(oyuncu_secimi, bilgisayar_secimi)
        son_oyun_zamani = time.time()
    if bilgisayar_secimi and (time.time() - son_oyun_zamani) < bekleme_suresi:
        cv.putText(frame, f"Bilgisayar: {bilgisayar_secimi}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.putText(frame, Sonuc, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        kalan = bekleme_suresi - (time.time() - son_oyun_zamani)
        cv.putText(frame, f"Yeni oyun {int(kalan)} saniye sonra", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    else:
        cv.putText(frame, "Elinizi gösterin (rock, paper, scissors)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.putText(frame, f"Skor - Oyuncu: {skor['oyuncu']} Bilgisayar: {skor['bilgisayar']}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)      
    
    cv.imshow("Rock Paper Scissors", frame)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        skor = {"oyuncu": 0, "bilgisayar": 0}
        son_oyun_zamani = 0
        bilgisayar_secimi = None
        Sonuc = None
        
cap.release()
cv.destroyAllWindows()

     
            
            
            
        
        
    