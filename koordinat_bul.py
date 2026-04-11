import cv2

# Tıkladığımız yerin koordinatlarını ekrana yazdıran fonksiyon
def fare_tiklamasi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Farenin sol tuşuna basıldığında
        print(f"[{x}, {y}],")

# Videomuzu açıyoruz
kamera = cv2.VideoCapture("test_video.mp4")

# Videonun sadece ilk karesini (fotoğrafını) okuyoruz
basarili_mi, ilk_kare = kamera.read()

if basarili_mi:
    # 1. PENCEREYİ SERBEST BOYUTLANDIRILABİLİR YAP (İşte sihirli kod bu)
    cv2.namedWindow("Koordinat Bulucu", cv2.WINDOW_NORMAL)
    
    # Fotoğrafı ekranda gösteriyoruz
    cv2.imshow("Koordinat Bulucu", ilk_kare)
    
    # Fare tıklamalarını dinlemeye başlıyoruz
    cv2.setMouseCallback("Koordinat Bulucu", fare_tiklamasi)
    
    print("\n--- KOORDİNAT BULUCU BAŞLADI ---")
    print("Lütfen videodaki yeşil alanın 4 köşesine (saat yönünde) sırayla tıklayın.")
    print("Tıkladığınız koordinatlar hemen aşağıda belirecektir.\n")
    
    # Ekranda kalmasını sağla, klavyeden bir tuşa basılana kadar bekle
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Video açılamadı, lütfen ismini kontrol edin.")