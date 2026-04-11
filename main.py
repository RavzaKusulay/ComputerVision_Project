import cv2
import numpy as np
import math
from ultralytics import YOLO
import winsound
from datetime import datetime
cv2.setUseOptimized(True)

print("Sistem başlatılıyor... Yapay zeka beyni yükleniyor...")

# Model
model = YOLO("best.pt")
sinif_isimleri = model.names
print("Model sınıfları:", sinif_isimleri)

kamera = cv2.VideoCapture("test_video.mp4")

# Güvenlik bölgesi
yasak_bolge = np.array([
    [1911, 1076],
    [1697, 1078],
    [1495, 372],
    [1561, 378]
], np.int32)

alarm_bekleme_suresi = 0

cv2.namedWindow("Akilli Sinir Guvenligi - Proje", cv2.WINDOW_NORMAL)

while True:
    ret, kare = kamera.read()
    if not ret:
        break

    # Güvenlik bölgesini çiz
    cv2.polylines(kare, [yasak_bolge], True, (255, 0, 0), 3)
    cv2.putText(kare, "GUVENLIK BOLGESI", (1505, 321),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 🔥 KRİTİK DÜZELTME
    sonuclar = model(kare, conf=0.1, imgsz=1280)

    ihlal_var = False

    for sonuc in sonuclar:
        for kutu in sonuc.boxes:

            x1, y1, x2, y2 = map(int, kutu.xyxy[0])
            sinif_id = int(kutu.cls[0])
            guven = round(float(kutu.conf[0]), 2)

            etiket = f"{sinif_isimleri[sinif_id]} {guven}"

            # Ayak noktası
            ayak_x = (x1 + x2) // 2
            ayak_y = y2

            cv2.circle(kare, (ayak_x, ayak_y), 4, (255, 0, 255), -1)

            # Bölge kontrolü
            iceride_mi = cv2.pointPolygonTest(
                yasak_bolge, (ayak_x, ayak_y), False
            )

            if iceride_mi >= 0:
                ihlal_var = True
                renk = (0, 0, 255)
            else:
                renk = (0, 255, 0)

            # Kutu çiz
            cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
            cv2.rectangle(kare, (x1, y1 - 30), (x2, y1), renk, -1)
            cv2.putText(kare, etiket, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    # Alarm ve log
    if ihlal_var:
        cv2.putText(kare, "ALARM: YASAK BOLGE IHLALI!",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 0, 255), 4)

        if alarm_bekleme_suresi == 0:
            winsound.Beep(1000, 500)

            zaman = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            with open("ihlal_log.txt", "a") as f:
                f.write(f"IHLAL: {zaman}\n")

            print("Ihlal kaydedildi:", zaman)
            alarm_bekleme_suresi = 30

    if alarm_bekleme_suresi > 0:
        alarm_bekleme_suresi -= 1

    cv2.imshow("Akilli Sinir Guvenligi - Proje", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()