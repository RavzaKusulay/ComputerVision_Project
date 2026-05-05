# Akıllı Sınır Güvenliği ve Tehlikeli Bölge İhlal Tespit Sistemi

Bu proje kapsamında derin öğrenme ve bilgisayarlı görü teknikleri kullanılarak gerçek zamanlı sınır güvenliği, tehlikeli bölge takibi ve otonom kanıt toplama yazılımı geliştirilmiştir. Geleneksel güvenlik kameralarının pasif kayıt mantığı aşılarak kameranın anlık tehdit analizi yapması, düşük ışıkta görüş yeteneğini otonom olarak iyileştirmesi ve izinsiz giriş yapan kişilerin fotoğraflarını arşivlemesi sağlanmıştır. Sistem mimarisinin arka planında YOLOv8 nesne tespit algoritmalarından ve ByteTrack çoklu nesne takibi yöntemlerinden faydalanılmıştır.

## Temel Özellikler

- **Gerçek Zamanlı Tehdit Tespiti ve Takip:** İzinsiz giriş yapan hedefler kesintisiz olarak takip edilmiş ve engellerin arkasına geçmeleri durumunda dahi kimlik numaralarının hafızada korunması sağlanmıştır.
- **Düşük Işık Optimizasyonu:** CLAHE ve Gaussian Blur algoritmaları kullanılarak düşük çözünürlüklü veya karanlık güvenlik kamerası görüntüleri yapay zekaya aktarılmadan önce otonom olarak aydınlatılmış ve kumlanmadan arındırılmıştır.
- **Otonom Kanıt Toplama:** Belirlenen tehlikeli bölgeye adım atan şüphelinin yüz veya beden görüntüsü ana video akışından otonom olarak kırpılmış ve zaman damgası ile kayıt klasörüne arşivlenmiştir.
- **Dinamik Gösterge Paneli:** Ekran üzerinde anlık sistem hızı, içerideki aktif tehdit sayısı ve toplam alarm sayısı canlı veri olarak sunulmuştur.
- **Çift Ekranlı Monitörizasyon:** Kullanıcıya aynı anda hem kameranın ham görüntüsü hem de yapay zekanın işlenmiş ve netleştirilmiş versiyonu senkronize olarak aktarılmıştır.

## Sistem Gereksinimleri ve Kurulum

Yazılımın yerel cihazlarda sorunsuz çalıştırılabilmesi için aşağıda belirtilen kütüphanelerin sistemde kurulu olması gerekmektedir. Gerekli bağımlılıkları indirmek adına terminal üzerinden aşağıdaki komutun çalıştırılması yeterlidir:

```bash
pip install opencv-python numpy ultralytics
```

## Kullanim ve Calistirma Adimlari

1. Egitilmis model agirlik dosyasi olan best.pt dosyasi proje ana dizinine eklenmelidir.
2. Test amaciyla kullanilacak guvenlik kamerasi video dosyasi proje klasorune dahil edilmelidir.
3. Gerekli dosya yollari kaynak kod icerisinde dogrulandiktan sonra sistem asagidaki komut ile baslatilmalidir:

```bash
python test_detect.py
```

## Proje Dizin Yapisi

Proje klasor hiyerarsisi asagidaki gibi duzenlenmistir:

- IHLAL_KAYITLARI: Sistem tarafindan otonom olusturulan supheli kanit fotograflarini barindirir.
- runs: YOLOv8 egitim ciktilarini ve model agirliklarini icerir.
- test_detect.py: Sistemin ana kaynak kod dosyasidir.
- main_test.mp4: Otonom analiz icin kullanilan ornek guvenlik kamerasi video dosyasidir.
- ihlal_log.txt: Tarih ve saat bazli metin tabanli olay gunlugudur.
- README.md: Proje dokumantasyon dosyasidir.
