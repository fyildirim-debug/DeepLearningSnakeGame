# Yapay Zeka Destekli Yılan Oyunu Projesi

## Proje Tanımı

Bu proje, yapay zeka (özellikle pekiştirmeli öğrenme - Reinforcement Learning, RL) kullanarak kendi kendine hareket etmeyi öğrenen, yaptığı hamleleri bir veritabanı benzeri yapıda kaydeden ve hatalarından ders alarak zamanla daha iyi performans gösteren akıllı bir yılan oyunu geliştirmeyi amaçlamaktadır.

## Kurallar ve Hedefler

*   [x] Yılan, oyun alanında rastgele beliren yemleri yiyerek büyümeli.
*   [x] Yılan kendi kendine veya oyun alanı sınırlarına çarpmamalı.
*   [x] Yapay zeka ajanı, zamanla en yüksek skoru elde etmek için en uygun hareket stratejisini öğrenmeli (Ödül fonksiyonu ve RL algoritması ile hedefleniyor).
*   [ ] Ajanın öğrenme süreci (hamleler, durumlar, ödüller) detaylı kaydedilmeli ve analiz edilebilmeli. (Şu an sadece model kaydediliyor.)
*   [x] Ajan, önceki deneyimlerinden ders çıkararak performansını iyileştirmeli (Model yükleme/kaydetme ile sağlandı).

## Teknoloji Seçimi

*   **Oyun Geliştirme:** Python, Pygame
*   **Yapay Zeka (RL):** Stable Baselines3 (DQN algoritması kullanılıyor)
*   **Veri Kaydı:** Model kaydı için Stable Baselines3'ün `.zip` formatı.

## Oluşturma Aşamaları (Yapılacaklar Listesi)

- [x] **1. Temel Yılan Oyunu Mekaniklerinin Geliştirilmesi:**
    - [x] Oyun alanı ve temel öğelerin (yılan, yem) çizilmesi.
    - [x] Yılanın hareket kontrolü (başlangıçta klavye ile).
    - [x] Yem yeme mekaniği ve yılanın büyümesi.
    - [x] Çarpışma tespiti (kendi kendine ve sınırlara).
    - [x] Skor hesaplama.
- [x] **2. Pekiştirmeli Öğrenme Ortamının Tanımlanması:**
    - [x] **Durum (State) Temsili:** Yem göreceli konumu, yön, yakın tehlikeler.
    - [x] **Aksiyon (Action) Alanı:** 4 yön (Yukarı, Aşağı, Sol, Sağ).
    - [x] **Ödül (Reward) Fonksiyonu:** Yem yeme (+50), çarpma (-100), yeme yaklaşma (+), zaman cezası (-).
- [x] **3. RL Algoritmasının Seçilmesi ve Entegrasyonu:**
    - [x] DQN algoritması seçildi ve Stable Baselines3 ile entegre edildi.
- [x] **4. Modelin Kaydedilmesi ve Geri Yüklenmesi:**
    - [x] Stable Baselines3'ün `model.save()` ve `model.load()` fonksiyonları kullanılarak eğitilmiş modelin `.zip` dosyası olarak kaydedilmesi ve yüklenmesi sağlandı. (Detaylı adım kaydı şimdilik yok.)
- [x] **5. Modelin Eğitilmesi ve Değerlendirilmesi:**
    - [x] RL ajanının oyun ortamında `model.learn()` ile eğitilmesi.
    - [x] Eğitim sürecinin terminal logları ile izlenmesi (SB3 logları).
    - [x] Eğitilmiş modelin performansının "Yapay Zeka ile Oyna" moduyla değerlendirilmesi.
- [x] **6. Öğrenilen Politikaların Kullanılarak Oyunun Oynatılması:**
    - [x] Eğitilmiş modelin yılanı kontrol etmesi ve oyunun izlenmesi sağlandı ("Yapay Zeka ile Oyna" modu).
- [x] **7. Kullanıcı Arayüzü ve Kontroller:**
    - [x] Oyun modlarını (Eğit, AI ile Oyna, Manuel Oyna, Çıkış) seçmek için terminal tabanlı menü eklendi.
    - [x] Eğitim sırasında oyunu duraklatma ('P') ve çıkma ('Q') özelliği eklendi (Render açıkken).
    - [x] Render kapalıyken eğitimi durdurup modeli kaydetmek için Ctrl+C kullanımı sağlandı.
- [x] **8. Görsel İyileştirmeler ve Ayarlar:**
    - [x] Oyun alanı boyutunun (`GRID_SIZE`) ayarlanabilmesi sağlandı.
    - [x] Oyun görsel hızının (`render_fps`) ayarlanabilmesi sağlandı.
    - [x] Eğitim bilgilerinin (skor, adım) oyun ekranında gösterilmesi sağlandı (Render açıkken).

## Neler Yapılabilir? (Gelecek Geliştirmeler)

*   Farklı RL algoritmalarının (PPO, A2C vb.) denenmesi ve karşılaştırılması.
*   Daha karmaşık durum temsilleri (örn. CNN için grid) veya ödül fonksiyonları denemek.
*   Eğitim sürecini daha detaylı görselleştiren araçlar (TensorBoard entegrasyonu).
*   Hiperparametre optimizasyonu (learning rate, buffer size vb.).
*   Engeller eklemek veya haritayı dinamikleştirmek.

## Karşılaşılan Hatalar ve Çözümler

*   **Hata:** `KeyError: 'terminateds'` (Stable Baselines3 versiyon uyumsuzluğu nedeniyle `SimpleInfoCallback` içinde).
    *   **Çözüm:** Özel `SimpleInfoCallback` sınıfı kaldırıldı, Stable Baselines3'ün kendi loglama mekanizmasına güvenildi.
*   **Hata:** Oyun menü yerine doğrudan başlıyordu.
    *   **Çözüm:** Eski test amaçlı `if __name__ == '__main__':` bloğu yorum satırına alındı.
*   **Sorun:** Render kapalıyken eğitim nasıl durdurulur ve model nasıl kaydedilir?
    *   **Çözüm:** Ctrl+C tuş kombinasyonunun `KeyboardInterrupt` oluşturduğu ve `train` fonksiyonundaki `try...finally` bloğunun modeli kaydettiği doğrulandı.
*   **Hata:** `TypeError: draw_menu() missing 4 required positional arguments: 'screen', 'width', 'height', and 'font'`
    *   **Çözüm:** `draw_menu` fonksiyonu Pygame'den bağımsız hale getirilerek terminal tabanlı basit bir menüye dönüştürüldü.

## Config Bilgileri (snake_game.py)

*   **Oyun Alanı:** `GRID_SIZE = 40`
*   **Blok Boyutu:** `block_size = 20` (Dolayısıyla ekran boyutu 800x800)
*   **Görsel Hız (FPS):** `metadata['render_fps'] = 15` (Oynama modları için)
*   **Eğitim Adım Sayısı:** `TIMESTEPS = 500000`
*   **Model Kayıt Yolu:** `MODEL_PATH = "dqn_snake_model.zip"`
*   **DQN Hiperparametreleri (Bazıları):**
    *   `learning_rate = 0.001`
    *   `buffer_size = 10000`
    *   `gamma = 0.99`
    *   `exploration_fraction = 0.2`
    *   `exploration_final_eps = 0.05`
*   **Ödül Fonksiyonu:**
    *   Yem yeme: +50
    *   Çarpma: -100
    *   Yaklaşma: +0.1 * (önceki_mesafe - yeni_mesafe)
    *   Zaman cezası: -0.1 (her adım) 