import cv2
import numpy as np
import math
from ultralytics import YOLO
import winsound
from datetime import datetime
import os 
import time 

# OpenCV kütüphanesinin bilgisayarın işlemcisini en verimli şekilde kullanmasını sağlar.
cv2.setUseOptimized(True)

print("Sistem başlatılıyor...")

# İhlal yapan kişilerin fotoğraflarının saklanacağı klasörü kontrol eder, yoksa yeni bir tane oluşturur.
kayit_klasoru = "IHLAL_KAYITLARI"
if not os.path.exists(kayit_klasoru):
    os.makedirs(kayit_klasoru)
    print(f"'{kayit_klasoru}' klasörü oluşturuldu.")

# Eğitilmiş yapay zeka modelini belleğe yükler.
model = YOLO("runs/detect/train/weights/best.pt")

print("Model yüklendi.")

# Kameradan gelen karanlık veya kumlu görüntüleri yapay zekanın daha net görebilmesi için işler.
def goruntu_iyilestir(orijinal_kare):
    # Görüntünün renk yapısını bozmadan sadece parlaklığını ayarlayabilmek için formatını dönüştürür.
    lab = cv2.cvtColor(orijinal_kare, cv2.COLOR_BGR2LAB)
    l_kanali, a_kanali, b_kanali = cv2.split(lab)

    # Karanlık noktaları bölgesel olarak aydınlatır ve detayları ortaya çıkarır.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_kanali)

    # Aydınlatılmış kanalı orijinal renklerle tekrar birleştirir.
    birlestirilmis_lab = cv2.merge((cl, a_kanali, b_kanali))
    iyilestirilmis = cv2.cvtColor(birlestirilmis_lab, cv2.COLOR_LAB2BGR)

    # Görüntüdeki hafif karıncalanmaları pürüzsüzleştirerek filtreler.
    iyilestirilmis = cv2.GaussianBlur(iyilestirilmis, (3, 3), 0)

    return iyilestirilmis

kamera = cv2.VideoCapture("main_test.mp4")

# Videonun başarıyla açılıp açılamadığını kontrol eder, açılamadıysa sistemi kapatır.
if not kamera.isOpened():
    print("Video açılamadı!")
    exit()

# Sistemin performansını korumak için her kareyi değil, belirtilen orandaki kareleri işlemesini sağlar.
kare_atlatma_orani = 2
kare_sayaci = 0

# Kameranın açısından anlık olarak çıkan kişilerin hafızada ne kadar süre tutulacağını belirler.
kutu_omru = 5

# Yapay zekanın verdiği rastgele kimlik numaralarını daha düzenli bir sıraya sokmak için sözlük oluşturur.
yolo_to_custom_ids = {}
next_id = 1

# Ekrandaki kişilerin konumlarını ve ihlal durumlarını aklında tutar.
hafiza_kutulari = {}
ihlal_eden_idler = set()

# Gösterge paneli için toplam ihlal sayısını sıfırdan başlatır ve sistemin çalışma hızını hesaplamak için zamanı tutar.
toplam_ihlal_sayisi = 0
fps_baslangic = time.time()

# Ekranda izinsiz giriş olarak kabul edilecek alanın köşe koordinatlarını belirler.
yasak_bolge = np.array([
    [1389, 4],
    [1550, 411],
    [1728, 1078],
    [1918, 1076],
    [1918, 454],
    [1476, 0]
], np.int32)

# Kullanıcıya gösterilecek olan video pencerelerinin boyutlandırılabilir olmasını sağlar.
cv2.namedWindow("1 - Ham Kamera (Orijinal)", cv2.WINDOW_NORMAL)
cv2.namedWindow("2 - Yapay Zeka Gorusu ve Alarmlar", cv2.WINDOW_NORMAL)

# Videodaki her bir kareyi bitene kadar tek tek okumak için sonsuz bir döngü başlatır.
while True:
    # Videodan anlık bir kare alır.
    ret, kare = kamera.read()

    # Eğer okunacak yeni bir kare kalmadıysa döngüyü sonlandırır.
    if not ret:
        break
        
    # Alınan ham kareyi yapay zeka için daha net bir hale getirir.
    islenmis_kare = goruntu_iyilestir(kare)

    # Kare sayacını günceller ve ekrandaki mevcut kişilerin listesini sıfırlar.
    kare_sayaci += 1
    mevcut_idler = set()
    
    # Anlık olarak ekranda bir ihlal olup olmadığını takip eder.
    ihlal_durumu = False
    
    # O an izinsiz giriş bölgesinde kaç kişi bulunduğunu sayar.
    anlik_icerideki_kisi = 0 

    # Sadece belirlenen aralıklardaki kareleri yapay zekaya göndererek sistemi hızlandırır.
    if kare_sayaci % kare_atlatma_orani == 0:
        
        # Yapay zeka modelini çalıştırarak ekrandaki kişileri tespit eder ve onlara bir kimlik numarası atar.
        sonuclar = model.track(
            islenmis_kare,
            persist=True,
            conf=0.5,
            iou=0.5,
            imgsz=640,
            tracker="bytetrack.yaml",
            stream=False
        )

        # Eğer ekranda herhangi bir nesne tespit edildiyse işlemlere başlar.
        if sonuclar and len(sonuclar) > 0:
            boxes = sonuclar[0].boxes

            # Tespit edilen nesnelerin sınırları ve kimlik numaraları mevcutsa döngüye girer.
            if boxes is not None and boxes.id is not None:
                kutular = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)

                # Tespit edilen her bir kişi için koordinatları ve kimlik numarasını eşleştirir.
                for kutu, yolo_id in zip(kutular, ids):
                    x1, y1, x2, y2 = map(int, kutu)
                    width = x2 - x1
                    height = y2 - y1

                    # Çok küçük boyutlardaki yanlış tespitleri dikkate almaz.
                    if width < 40 or height < 80:
                        continue

                    # Nesnenin insan boyutlarına uygunluğunu kontrol eder, uygun değilse eler.
                    oran = height / width
                    if oran < 1.2:
                        continue

                    # Kişiye atanan eski karmaşık numarayı sistemin kendi sırasına göre numaralandırır.
                    if yolo_id not in yolo_to_custom_ids:
                        yolo_to_custom_ids[yolo_id] = next_id
                        next_id += 1

                    custom_id = yolo_to_custom_ids[yolo_id]
                    mevcut_idler.add(custom_id)

                    # Kişinin konumunu ve ekranda kalma süresini hafızaya kaydeder.
                    hafiza_kutulari[custom_id] = [
                        [x1, y1, x2, y2],
                        kutu_omru
                    ]

    silinecekler = []

    # Hafızadaki her bir kişi için güncel durum analizi yapar.
    for s_id, veri in hafiza_kutulari.items():
        
        # Eğer kişi ekrandan çıktıysa hafızadaki ömrünü yavaşça azaltır.
        if s_id not in mevcut_idler and kare_sayaci % kare_atlatma_orani == 0:
            hafiza_kutulari[s_id][1] -= 1
            
        # Eğer kişi hala ekrandaysa hafıza ömrünü yeniler.
        elif s_id in mevcut_idler:
            hafiza_kutulari[s_id][1] = kutu_omru

        # Hafıza ömrü dolan kişileri sistemden tamamen silmek için işaretler.
        if hafiza_kutulari[s_id][1] <= 0:
            silinecekler.append(s_id)
            continue

        x1, y1, x2, y2 = veri[0]
        
        # Kişinin tam olarak ayaklarının bastığı orta noktayı hesaplar.
        ayak_x = (x1 + x2) // 2
        ayak_y = y2

        # Kişinin ayak noktasının, tanımlanan izinsiz giriş bölgesinin içinde olup olmadığını matematiksel olarak kontrol eder.
        iceride_mi = cv2.pointPolygonTest(
            yasak_bolge,
            (ayak_x, ayak_y),
            False
        )

        # Eğer kişi izinsiz giriş bölgesinin içindeyse alarm durumuna geçer.
        if iceride_mi >= 0:
            renk = (0, 0, 255)
            ihlal_durumu = True
            anlik_icerideki_kisi += 1 

            # Eğer bu kişi daha önce ihlal listesine eklenmediyse yeni bir kayıt oluşturur.
            if s_id not in ihlal_eden_idler:
                ihlal_eden_idler.add(s_id)
                toplam_ihlal_sayisi += 1 
                
                # İhlal anında bir bilgisayar uyarı sesi çıkarır.
                winsound.Beep(1000, 300)

                zaman = datetime.now().strftime("%d-%m-%Y %H-%M-%S") 
                
                # Kişinin yüzünü ve bedenini ana videodan kırparak bir fotoğraf haline getirir.
                kirpma_y1 = max(0, y1)
                kirpma_y2 = min(kare.shape[0], y2)
                kirpma_x1 = max(0, x1)
                kirpma_x2 = min(kare.shape[1], x2)
                kisi_fotografi = kare[kirpma_y1:kirpma_y2, kirpma_x1:kirpma_x2]
                
                # Kırpılan fotoğrafı hata oluşmasını engelleyerek ilgili klasöre kaydeder.
                if kisi_fotografi.size > 0:
                    dosya_adi = os.path.join(kayit_klasoru, f"Ihlal_ID{s_id}_{zaman}.jpg")
                    cv2.imwrite(dosya_adi, kisi_fotografi)

                # İhlal detaylarını saat ve tarih bilgisiyle beraber metin dosyasına yazar.
                zaman_log = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                with open("ihlal_log.txt", "a") as f:
                    f.write(f"IHLAL: ID {s_id} - {zaman_log}\n")

        # Eğer kişi güvenli bölgedeyse sistemi normal işleyişinde tutar.
        else:
            renk = (0, 255, 0)

        # Kişinin etrafına durumuna uygun renkte bir dikdörtgen çizer.
        cv2.rectangle(
            islenmis_kare,
            (x1, y1),
            (x2, y2),
            renk,
            2
        )

        # Çizilen dikdörtgenin hemen üstüne kişinin kimlik numarasını yazar.
        cv2.putText(
            islenmis_kare,
            f"ID:{s_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            renk,
            2
        )

    # Ekrandan uzun süre önce çıkmış kişileri hafıza sözlüğünden kalıcı olarak siler.
    for s in silinecekler:
        if s in hafiza_kutulari:
            del hafiza_kutulari[s]

    # Ekrana izinsiz giriş yapılacak olan o belirlenmiş alanı mavi renkle çizer.
    cv2.polylines(
        islenmis_kare,
        [yasak_bolge],
        True,
        (255, 0, 0),
        3
    )
    
    cv2.putText(islenmis_kare, "IZINSIZ GIRIS BOLGESI", (1290, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    if ihlal_durumu:
        cv2.putText(islenmis_kare, "UYARI! IZINSIZ GIRIS TESPIT EDILDI", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
    # Ekranın sol üst köşesine gösterge panelinin arka planı olacak yarı saydam siyah bir kutu çizer.
    cv2.rectangle(islenmis_kare, (20, 80), (450, 220), (0, 0, 0), -1) 
    
    # Sistemin bir saniyede kaç kare işlediğini hesaplar.
    fps_bitis = time.time()
    fps = 1 / (fps_bitis - fps_baslangic)
    fps_baslangic = fps_bitis
    
    # Gösterge panelinin üzerine sistemin anlık hızını, içerideki tehdit sayısını ve toplam ihlal miktarını yazar.
    cv2.putText(islenmis_kare, f"SISTEM HIZI: {int(fps)} FPS", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(islenmis_kare, f"ICERIDEKI TEHDIT: {anlik_icerideki_kisi}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if ihlal_durumu else (0, 255, 0), 2)
    cv2.putText(islenmis_kare, f"TOPLAM ALARM: {toplam_ihlal_sayisi}", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Hem orijinal kamerayı hem de yapay zeka görüntüsünü büyük ekranlarda daha net incelenebilmesi için genişletir.
    dev_orijinal = cv2.resize(kare, (1280, 720))
    dev_islenmis = cv2.resize(islenmis_kare, (1280, 720))

    # Genişletilmiş bu iki görüntüyü birbirinden bağımsız iki ayrı pencere halinde ekrana yansıtır.
    cv2.imshow("1 - Ham Kamera (Orijinal)", dev_orijinal)
    cv2.imshow("2 - Yapay Zeka Gorusu ve Alarmlar", dev_islenmis)

    # Kullanıcı klavyeden 'q' tuşuna basarsa tüm sistemi güvenli bir şekilde kapatır.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Arka planda çalışan kamera bağlantısını keser ve açık kalan tüm pencereleri kapatarak programı sonlandırır.
kamera.release()
cv2.destroyAllWindows()