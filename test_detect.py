import cv2
import numpy as np
import math
from ultralytics import YOLO
import winsound
from datetime import datetime

cv2.setUseOptimized(True)

print("Sistem başlatılıyor...")

# MODEL YÜKLE
model = YOLO("runs/detect/train/weights/best.pt")

print("Model yüklendi.")

kamera = cv2.VideoCapture("main_test.mp4")

if not kamera.isOpened():
    print("Video açılamadı!")
    exit()

# FPS hızlandırma
kare_atlatma_orani = 2
kare_sayaci = 0

# Bounding box hafıza süresi
kutu_omru = 5

# ID sistemi
yolo_to_custom_ids = {}
next_id = 1

hafiza_kutulari = {}
ihlal_eden_idler = set()

# Yasak bölge
yasak_bolge = np.array([
    [1389, 4],
    [1550, 411],
    [1728, 1078],
    [1918, 1076],
    [1918, 454],
    [1476, 0]
], np.int32)

cv2.namedWindow("AKILLI GUVENLIK", cv2.WINDOW_NORMAL)

while True:

    ret, kare = kamera.read()

    if not ret:
        break

    kare_sayaci += 1
    mevcut_idler = set()

    # Her 2 karede bir AI çalıştır
    if kare_sayaci % kare_atlatma_orani == 0:

        sonuclar = model.track(
            kare,
            persist=True,
            conf=0.5,
            iou=0.5,
            imgsz=640,
            tracker="bytetrack.yaml",
            stream=False
        )

        if sonuclar and len(sonuclar) > 0:

            boxes = sonuclar[0].boxes

            if boxes is not None and boxes.id is not None:

                kutular = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)

                for kutu, yolo_id in zip(kutular, ids):

                    x1, y1, x2, y2 = map(int, kutu)

                    width = x2 - x1
                    height = y2 - y1

                    # Küçük yanlış kutuları sil
                    if width < 40 or height < 80:
                        continue

                    # İnsan oranı filtresi
                    oran = height / width

                    if oran < 1.2:
                        continue

                    # Sabit ID sistemi
                    if yolo_id not in yolo_to_custom_ids:

                        yolo_to_custom_ids[yolo_id] = next_id
                        next_id += 1

                    custom_id = yolo_to_custom_ids[yolo_id]

                    mevcut_idler.add(custom_id)

                    hafiza_kutulari[custom_id] = [
                        [x1, y1, x2, y2],
                        kutu_omru
                    ]

    silinecekler = []

    for s_id, veri in hafiza_kutulari.items():

        if s_id not in mevcut_idler and kare_sayaci % kare_atlatma_orani == 0:

            hafiza_kutulari[s_id][1] -= 1

        elif s_id in mevcut_idler:

            hafiza_kutulari[s_id][1] = kutu_omru

        if hafiza_kutulari[s_id][1] <= 0:

            silinecekler.append(s_id)
            continue

        x1, y1, x2, y2 = veri[0]

        ayak_x = (x1 + x2) // 2
        ayak_y = y2

        # Yasak bölge kontrol
        iceride_mi = cv2.pointPolygonTest(
            yasak_bolge,
            (ayak_x, ayak_y),
            False
        )

        if iceride_mi >= 0:

            renk = (0, 0, 255)

            if s_id not in ihlal_eden_idler:

                ihlal_eden_idler.add(s_id)

                winsound.Beep(1000, 300)

                zaman = datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )

                with open("ihlal_log.txt", "a") as f:

                    f.write(
                        f"IHLAL: ID {s_id} - {zaman}\n"
                    )

        else:

            renk = (0, 255, 0)

        cv2.rectangle(
            kare,
            (x1, y1),
            (x2, y2),
            renk,
            2
        )

        cv2.putText(
            kare,
            f"ID:{s_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            renk,
            2
        )

    for s in silinecekler:

        if s in hafiza_kutulari:

            del hafiza_kutulari[s]

    cv2.polylines(
        kare,
        [yasak_bolge],
        True,
        (255, 0, 0),
        3
    )

    cv2.imshow(
        "AKILLI GUVENLIK",
        kare
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

kamera.release()
cv2.destroyAllWindows()