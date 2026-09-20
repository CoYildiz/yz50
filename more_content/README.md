# more_content — Perrotta, *Programming Machine Learning*

YZ50'nin haftalık görevlerinin **dışında**, paralel yürüyen ekstra okuma. Teslim edilmiyor.

**Kaynak:** Paolo Perrotta, *Programming Machine Learning*. Educative'deki sürümü:
[Fundamentals of Machine Learning for Software Engineers](https://www.educative.io/courses/fundamentals-of-machine-learning-for-software-engineers).

**Neden bu kitap, YZ50 varken:** Karpathy hızlı gidiyor ve PyTorch'a erken geçiyor. Perrotta aynı
yolu saf NumPy'la, adım adım ve her adımın gerekçesiyle yürüyor. İkisi çakışmıyor, üst üste biniyor:
burada yavaşça kurulan şey (lineer regresyon → matris gradyanı → sigmoid → log loss → MNIST),
YZ50'de daha hızlı ve daha büyük ölçekte tekrar karşına çıkıyor.

Dosyalar kitabın bölüm sırasını takip ediyor: `A` → `B` → `C` → `D` → `E`.

---

## Dosyalar

### `A_basic_lin_reg.py` — Bölüm 2-3: ilk öğrenen program

Tek değişkenli lineer regresyon, kütüphane yok. Rezervasyon sayısından satılan pizzayı tahmin ediyor
(`data/pizza.txt`, 30 örnek).

- `predict(X, w, b)`, `loss()` (MSE), `gradient()`, `train()` — gradient descent döngüsü elle
- `w` ve `b` **ayrı iki parametre**, türevleri de ayrı ayrı yazılıyor (`w_gradient`, `b_gradient`)
- Sonunda **kapalı form** (normal denklemler) ile aynı sonuca varılıyor, iki yöntem karşılaştırılıyor

**Buradan çıkan asıl ders** — dosyanın sonundaki yorumda da duruyor: kapalı form formülünün
**ortalama** hali ile **toplam** hali karıştırılırsa `n` çarpanları yalnızca bir terime uygulanıyor
ve eğim sessizce şişiyor (1.08 yerine 1.84). Hata vermiyor, sadece yanlış cevap veriyor.

Çıktılar: `outputs/basic_lin_reg/`

### `B_more_dimension.py` — Bölüm 4: hiperuzay

Üç girdi değişkeni (rezervasyon, sıcaklık, turist → pizza, `data/pizza_3_vars.txt`). Kod döngüden
matris çarpımına iniyor.

- **Bias hilesi:** `X`'in başına hep `1` olan bir sütun ekleniyor, böylece `b` ayrı bir değişken
  olmaktan çıkıp `w[0]` oluyor. `X @ w` tek işlemde bias'ı da içeriyor
- Gradyan tek satıra iniyor: `2/m * X.T @ (tahmin - y)`. Ayrı bir `b_gradient` yok — `X.T`'nin ilk
  satırı hep `1` olduğu için bias'ın türevi çarpımdan kendiliğinden düşüyor
- 3B grafik: veri noktaları ve oturan düzlem

**Bulunan iki hata (düzeltildi):**
1. Grafik `train()` çağrılmadan önce çiziliyordu — kaydedilen görüntü sıfır ağırlıklı, düz bir
   levhaydı. Perrotta'nın orijinalinde o noktada eğitimden çıkmış ağırlıklar elle yazılı olduğu için
   fark edilmiyor
2. Eksen etiketleri veri sütunlarına göre tersti (`x1` rezervasyon, ama eksende "Temperature"
   yazıyordu)

**Ortak ders:** matplotlib hata vermez. Yanlış ağırlıkla da, yanlış etiketle de bir resim üretir.
Grafik bir doğrulama aracıysa, önce grafiğin kendisi doğrulanmalı.

Çıktılar: `outputs/more_dimensions/`

### `C_deep_learningwtorch.py` — boş

Yer tutucu. PyTorch karşılaştırması için açıldı, henüz yazılmadı.

### `D_discern_mach.py` — Bölüm 5: sınıflandırmaya geçiş

Aynı pizza verisi, ama soru değişiyor: sayı tahmin etmek yerine **evet/hayır** (`data/police.txt` —
o gün polis geldi mi).

Modelde üç şey değişiyor:

1. **Sigmoid** ekleniyor: `forward()` artık `sigmoid(X @ w)` döndürüyor, çıktı `(0, 1)` aralığında
2. Ayrı bir **`classify()`** var: `forward()`'ın sürekli çıktısını en yakın tam sayıya yuvarlıyor.
   Eğitimde yumuşak çıktı gerekiyor (gradient descent'in üzerinde kayacağı yüzey), sınıflandırmada
   kesin cevap
3. **Loss MSE'den log loss'a geçiyor.** Sebep: MSE'yi sigmoid'le birlikte tutarsan loss yüzeyi
   yerel minimumlarla doluyor ve gradient descent çukurlarda takılıyor

`predict()` adı `forward()` oldu — veriyi sistemin içinden ileri geçirme işleminin adı
**forward propagation**, ve bu isim buradan sonra her yerde kullanılıyor.

### `E_mnist_classfier_for5.py` — Bölüm 6: gerçek veri

Aynı ikili sınıflandırıcı, bu sefer MNIST üzerinde: **"bu rakam 5 mi?"**

- `prepare_X()` — her `28×28` görüntüyü tek satıra düzleştiriyor (784 sayı), başa bias sütunu
  ekliyor → `(60000, 785)`. Her piksel bir özellik
- `encode_labels()` — MNIST etiketleri `0-9`, model ikili. `(labels == 5)` ile `0/1`'e çevriliyor.
  `reshape(-1, 1)` **şart**: `(m,)` bırakılırsa `forward(X,w) - Y` broadcasting yüzünden `(m, m)`
  üretir, hata vermeden yanlış çalışır

**Sonuç:** test setinde %96.78. Loss `0.693`'ten (= `ln(2)`, hiçbir şey bilmeyen modelin değeri)
`0.109`'a indi.

**`lr = 1e-5` neden bu kadar küçük:** piksel değerleri `0-255` ölçeğinde bırakıldı. 784 tane böyle
sayı ağırlıklarla çarpılınca learning rate tavanı çok aşağı iniyor. `1e-4` denersen `nan` görürsün.

**Düzleştirmenin sessiz bedeli:** ızgarada komşu olan iki piksel (`(5,10)` ve `(6,10)`) satırda 28
eleman uzağa düşüyor. Model bu ikisinin komşu olduğunu bilmiyor, herhangi iki sütun gibi davranıyor.
CNN'in varlık sebebi bu.

### `F_digit_error_analysis.py` — Bölüm 6'ya ek analiz

Kitap "hangi rakam zor" sorusunu sorup cevabını accuracy tablosuyla veriyor, sonra "isteyen daha
derine inebilir" diyerek bırakıyor. Bu dosya oraya iniyor: 10 rakamın her biri için ayrı bir ikili
sınıflandırıcı eğitiyor ve **hataları ayrıştırıyor**.

```
false negative (FN) : gerçek rakam BU, model "hayır" dedi      -> kaçırdı
false positive (FP) : gerçek rakam BAŞKA, model "evet" dedi    -> karıştırdı
```

**Birinci bulgu — accuracy yanıltıcı.** Her sınıflandırıcı "bu bir 8 mi" diye soruyor ve test
setinin yalnızca ~%10'u 8. Yani hiçbir şey öğrenmeyip her örneğe "hayır" diyen model bile ~%90
alıyor. Başlangıç çizgisi %50 değil, %90:

| Rakam | Adet | "Hep hayır" der | Model | Kazanç | FN | FP | **Recall** |
|---|---|---|---|---|---|---|---|
| 0 | 980 | 90.20% | 98.99% | +8.79 | 49 | 52 | 95.0% |
| 1 | 1135 | 88.65% | 99.03% | **+10.38** | 64 | 33 | 94.4% |
| 2 | 1032 | 89.68% | 97.37% | +7.69 | 212 | 51 | 79.5% |
| 3 | 1010 | 89.90% | 96.98% | +7.08 | 220 | 82 | 78.2% |
| 4 | 982 | 90.18% | 97.59% | +7.41 | 177 | 64 | 82.0% |
| 5 | 892 | 91.08% | 96.37% | +5.29 | 295 | 68 | 66.9% |
| 6 | 958 | 90.42% | 98.07% | +7.65 | 114 | 79 | 88.1% |
| 7 | 1028 | 89.72% | 98.14% | +8.42 | 138 | 48 | 86.6% |
| **8** | 974 | 90.26% | 93.85% | **+3.59** | **419** | 196 | **57.0%** |
| 9 | 1009 | 89.91% | 95.57% | +5.66 | 300 | 143 | 70.3% |

Ham accuracy'de "93.85 vs 99.03" küçük bir fark gibi duruyor. Kazanca bakınca 8 sınıflandırıcısı
hiçbir şey yapmamaya göre **3.6 puan** kazanmış, 1 sınıflandırıcısı **10.4 puan**.

**İkinci bulgu — asıl gizlenen sayı recall.** 8 için accuracy %93.85 ama **recall %57**: model
gerçek 8'lerin ancak yarısından biraz fazlasını yakalıyor. Accuracy bunu gizliyor çünkü doğru
bildiği 9026 tane "8 değil" örneği paydayı şişiriyor.

**Üçüncü bulgu — hangi hata baskın.** 419 FN'e karşı 196 FP: model 8'leri başka rakamlarla
**karıştırmıyor**, 8'leri **bulamıyor**. Yanlış "evet" dediklerinin dağılımı: 5 (73 kez), 2 (46),
1 (30). 5 ↔ 8 çift yönlü karışıyor — 5'in kendi recall'u da %66.9 ile sondan ikinci.

**Neden 8 zor, 1 kolay:** model tek katmanlı ve ham piksellerle çalışıyor, öğrendiği şey pratikte
bir **şablon**. `1` ince, dikey, hep aynı bölgeden geçen bir çizgi — çok ayırt edici. `8` neredeyse
bütün kareyi dolduruyor ve 5, 2, 3 ile piksel düzeyinde büyük örtüşme var; doğrusal bir sınır
bunları ayıramıyor. Kitabın "insan 7 ve 4 zor olur sanır, yanlış" gözleminin sebebi bu: doğrusal
model için önemli olan el yazısı çeşitliliği değil, **diğer sınıflarla piksel örtüşmesi**.

**Genel ders:** dengesiz veride accuracy neredeyse hiçbir şey söylemez. Her zaman iki soruyu sor:
*hiçbir şey yapmayan model ne alırdı*, ve *recall kaç*.

---

## Veri

| Dosya | Ne | Nerede kullanılıyor |
|---|---|---|
| `data/pizza.txt` | 30 örnek, rezervasyon → pizza | `A` |
| `data/pizza_3_vars.txt` | 30 örnek, 3 girdi → pizza | `B` |
| `data/police.txt` | 30 örnek, 3 girdi → polis geldi mi (0/1) | `D` |
| MNIST | 60.000 eğitim + 10.000 test | `E`, `F` |

MNIST `load_dataset("ylecun/mnist")` ile indiriliyor ve **repoya dahil değil** — parquet dosyaları
18 MB, `.gitignore`'da. İlk çalıştırmada kendiliğinden iniyor.

## Çalıştırma

```bash
uv run python more_content/A_basic_lin_reg.py
```

Dosyalar proje kökünden çalıştırılmak üzere yazıldı (veri yolları `data/...` biçiminde göreli).

## İlgili notlar

Buradaki konuların ayrıntılı notları vault tarafında, `RoadMapsProjects/yz50/notlar/` altında:

- `01-lineer-regresyon` — `A`'nın konusu: normal denklemler, koşullanma, MSE'nin MLE olması
- `02-hiperuzay` — `B`'nin konusu: bias hilesi, matris gradyanı, özellik ölçekleri
- `03-siniflandirmaya-gecis` — `D`'nin konusu: sigmoid, log loss, ve accuracy'nin neden optimize
  edilemediğinin ispatı; `F`'nin bulguları da burada
- `15-loss-nll-cross-entropy` — loss'un nereden geldiği (maximum likelihood) ve log loss'un sayısal
  kararlılığı
