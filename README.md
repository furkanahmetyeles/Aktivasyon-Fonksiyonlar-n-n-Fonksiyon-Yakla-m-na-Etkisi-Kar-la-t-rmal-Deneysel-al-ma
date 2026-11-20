# Aktivasyon Fonksiyonlarının Fonksiyon Yaklaşımına Etkisi: Bir Karşılaştırmalı Deneysel Çalışma

Bu depo, Manisa Celal Bayar Üniversitesi Yapay Sinir Ağları dersi kapsamında Springer *Emerging Trends in Mechatronics* serisi için hazırlanan kitap bölümünün deneysel kodlarını ve veri işleme adımlarını içermektedir.

## Yazarlar
1. Danışman Hoca (Manisa Celal Bayar Üniversitesi)
2. Furkan Ahmet Yeles (Manisa Celal Bayar Üniversitesi, Makine Mühendisliği Bölümü)

## Veri Seti
- **Adı:** UCI Energy Efficiency Dataset (ID: 242)
- **Kaynak:** https://archive.ics.uci.edu/dataset/242/energy+efficiency
- **Özellikler:** 768 bina simülasyonu örneği, 8 giriş özniteliği ($X_1$–$X_8$) ve 2 sürekli çıkış hedefi ($Y_1$: Isıtma Yükü, $Y_2$: Soğutma Yükü).

## Kurulum ve Çalıştırma
1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. Deney betiğini veya Jupyter Notebook dosyasını çalıştırın:
   ```bash
   python deney_calistir.py
   ```
   veya `YSA_Aktivasyon_Fonksiyonlari_Deneyler.ipynb` dosyasını Jupyter/Colab ortamında çalıştırın.

## Çıktılar
- `figures/` klasörü altında 300 DPI çözünürlüklü 7 adet PNG grafik (`G1`–`G7`).
- `results/` klasörü altında tüm tablo verileri (`T1`, `T6`, `T7`, `T8`, `T10` CSV dosyaları ve `tum_deney_sonuclari.json`).
