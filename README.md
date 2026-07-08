# Airport Scene Risk Analysis

Bu proje, havaalanı görüntülerini analiz ederek hem **semantic risk** (alan tipi bazlı risk) hem de **operational risk** (operasyonel yoğunluk/anomali) belirlemeyi amaçlar. Sonuç olarak, her havaalanı sahnesi için detaylı bir **risk heatmap** ve patch bazlı risk skorları üretir.

---

## 📌 Proje Amacı

Havaalanları, uçak, apron, pist ve terminal gibi operasyonel alanlarıyla karmaşık yapılara sahiptir. Bu projede amaç:

1. Görüntülerden **airport component** ve **non-airport** alanlarını ayırmak.
2. **Semantic Risk** hesaplamak: Airport dışı alanlar (ağaç, toprak, şehir vb.) riskli, airport component alanlar güvenli kabul edilir.
3. **Operational Risk** hesaplamak: Airport component alanlar içinde Dual AutoEncoder kullanılarak yapısal anomali ve sapmalar belirlenir.
4. Sonuçları kullanıcı dostu bir **Streamlit Web Arayüzü** üzerinden görselleştirmek.

---

## ⚙️ Kullanılan Modeller

1. **Binary Semantic Classifier (ResNet18)**
   - Airport component vs non-airport ayrımı yapar.
   - Çıktı, her patch için **airport_component_probability** ve **non_airport_probability** değerlerini sağlar.
   - Başarı metrikleri (terminalde hesaplandı):
     - Accuracy: 0.9887
     - F1 Score: 0.9882
     - Precision: 0.9941
     - Recall: 0.9825
     - IoU: 0.9767
   - Neden: Semantic risk için yüksek doğruluklu, hızlı ve güvenilir patch sınıflandırması sağlar.

2. **Dual AutoEncoder**
   - Airport component patchleri üzerinde çalışır.
   - Yapısal sapmaları ve anomaliyi tespit ederek **operational risk heatmap** üretir.
   - Neden: Patch-level anomali tespiti için denetimsiz ve güvenilir bir yöntemdir.

---

## 🧩 Proje Mantığı

- **Semantic Heatmap**:
  - Patch'ler üzerinde binary semantic classifier uygulanır.
  - Airport component alanlar güvenli, non-airport alanlar riskli kabul edilir.
  - Heatmap renkleri:  
    - Mavi = Güvenli (airport component)  
    - Kırmızı = Riskli (non-airport)

- **Operational Heatmap**:
  - Dual AutoEncoder ile reconstruction error hesaplanır.
  - Patch bazlı hata değeri operational risk skoruna dönüştürülür.

- **Final Risk Heatmap**:
  - İki heatmap ağırlıklı birleştirilir:  
    ```
    Final Risk = α * Semantic Risk + β * Operational Risk
    ```
  - α = 0.85, β = 0.15 (öncelik semantic risk).

---

## 📊 Başarı Metrikleri

- **Semantic Risk (Binary Semantic Classifier)**
  - Accuracy: 0.9887
  - F1 Score: 0.9882
  - Precision: 0.9941
  - Recall: 0.9825
  - IoU: 0.9767

- **Operational Risk (Dual AutoEncoder)**
  - Reconstruction error üzerinden raporlanır, anomali tespiti amaçlıdır.

- **Global Risk Score**
  - Semantic + Operational risk kombinasyonu ile elde edilir.

---

## 💻 Kurulum ve Kullanım

1. Python 3.10+ gereklidir.
2. Gerekli paketler:  
```bash
pip install -r requirements.txt

## Proje Görüntüsü

<img width="1714" height="843" alt="Ekran görüntüsü 2026-05-24 233013" src="https://github.com/user-attachments/assets/f56d5893-e77c-4c5e-9e41-57df4b72a058" />


