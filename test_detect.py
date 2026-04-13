import cv2
import numpy as np
import math
from ultralytics import YOLO
import winsound
from datetime import datetime

cv2.setUseOptimized(True)

print("Sistem başlatılıyor... Yapay zeka beyni yükleniyor...")

model = YOLO("yolov8n.pt")
sinif_isimleri = model.names
print("Model sınıfları yüklendi.")

kamera = cv2.VideoCapture("main_test.mp4")

yasak_bolge = np.array([
[1389, 4],
[1550, 411],
[1728, 1078],
[1918, 1076],
[1918, 454],
[1476, 0],
], np.int32)

ihlal_eden_idler = set()

# 🔥 YENİ: MEKANSAL HAFIZA SİSTEMİ
son_gorulen_konumlar = {} # Hangi ID en son nerede görüldü? (X, Y)
id_donusum_tablosu = {}   # YOLO'nun verdiği saçma ID'leri, bizim gerçek ID'lerimize bağlar

cv2.namedWindow("Akilli Sinir Guvenligi - Proje", cv2.WINDOW_NORMAL)

while True:
    ret, kare = kamera.read()
    if not ret: break

    sonuclar = model.track(kare, persist=True, conf=0.45, iou=0.5)

    for sonuc in sonuclar:
        if sonuc.boxes.id is not None:
            kutular = sonuc.boxes.xyxy.cpu().numpy()
            ids = sonuc.boxes.id.cpu().numpy().astype(int)
            siniflar = sonuc.boxes.cls.cpu().numpy().astype(int)

            for kutu, yolo_id, sinif_id in zip(kutular, ids, siniflar):
                if sinif_id == 0:
                    
                    x1, y1, x2, y2 = map(int, kutu)
                    ayak_x = int((x1 + x2) // 2)
                    ayak_y = int(y2)

                    # 🔥 AKILLI ID EŞLEŞTİRME (Mekansal Hafıza)
                    gercek_id = yolo_id # Varsayılan olarak YOLO'nun ID'sini kabul et

                    # 1. Eğer bu YOLO ID'sini daha önce kendi ID'mize çevirdiysek, onu kullan
                    if yolo_id in id_donusum_tablosu:
                        gercek_id = id_donusum_tablosu[yolo_id]
                    else:
                        # 2. Yeni bir ID geldi. Acaba eski kaybolanlardan birine YAKIN MI?
                        for eski_id, (eski_x, eski_y) in son_gorulen_konumlar.items():
                            # İki nokta arasındaki mesafeyi ölçüyoruz
                            mesafe = math.hypot(ayak_x - eski_x, ayak_y - eski_y)
                            
                            # Eğer 150 pikselden daha yakın bir yerde belirdiyse, bu O'dur!
                            if mesafe < 150: 
                                id_donusum_tablosu[yolo_id] = eski_id # Listeye kaydet
                                gercek_id = eski_id # ID'yi değiştir
                                break
                    
                    # Adamın son konumunu hafızaya kaydet/güncelle
                    son_gorulen_konumlar[gercek_id] = (ayak_x, ayak_y)

                    # Bölge kontrolü
                    iceride_mi = cv2.pointPolygonTest(yasak_bolge, (ayak_x, ayak_y), False)

                    if iceride_mi >= 0:
                        renk = (0, 0, 255) 
                        
                        # 🔥 Artık "gercek_id" üzerinden kontrol yapıyoruz
                        if gercek_id not in ihlal_eden_idler:
                            ihlal_eden_idler.add(gercek_id)
                            
                            winsound.Beep(1000, 500)
                            zaman = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                            
                            with open("ihlal_log.txt", "a") as f:
                                f.write(f"IHLAL: ID {gercek_id} - Zaman: {zaman}\n")
                            
                            print(f"ALARM! Yeni İhlal -> ID: {gercek_id} | Saat: {zaman}")
                    else:
                        renk = (0, 255, 0)

                    # Çizimleri Yap (Ekrana "gercek_id"yi yazdırıyoruz)
                    etiket = f"ID:{gercek_id} {sinif_isimleri[sinif_id]}"
                    cv2.circle(kare, (ayak_x, ayak_y), 4, (255, 0, 255), -1) 
                    cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
                    cv2.rectangle(kare, (x1, y1 - 30), (x2, y1), renk, -1)
                    cv2.putText(kare, etiket, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.polylines(kare, [yasak_bolge], True, (255, 0, 0), 3)
    cv2.putText(kare, "TEHLIKELI BOLGE", (1362, 1031), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Akilli Sinir Guvenligi - Proje", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()