# Hafta 4 — İlk Neural Network Language Model (MLP, Bengio 2003)

**Teslim tarihi:** 13 Eylül 2026

## Bu hafta öğrenilecekler
- Embedding: harfi one-hot yerine öğrenilen küçük bir vektörle temsil etmek
- Bağlam penceresi: bir harf yerine önceki üç harfe bakan model
- Minibatch ile eğitim ve learning rate seçimi
- Train / dev / test ayrımı ve neden gerektiği
- Tanh saturation: aktivasyonlar ±1'e yığılınca gradient'in ölmesi
- Kaiming init: ağırlıkları hangi ölçekte başlatmak gerektiği
- BatchNorm: aktivasyonları katman katman normalize etmek ve eğitime etkisi

## Kaynaklar
- [Karpathy — Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
- [Karpathy — Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- [Bengio 2003 — A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — bu haftanın modelinin makalesi
- [Karpathy'nin makemore reposu](https://github.com/karpathy/makemore)

## Görevler
- [x] Önceki üç harfi bağlam alan veri setini kur (X: 3 harf indeksi, Y: sıradaki harf), embedding tablosunu oluştur
- [x] Gizli katman + çıkış katmanı: embedding'leri düzleştir, `W1`/`b1` ile tanh, `W2`/`b2` ile logits; loss'u `F.cross_entropy` ile doğrula
- [x] Eğitim döngüsü: minibatch'lerle eğit, learning rate'i tara, train/dev/test böl, loss'u dev üzerinde raporla
- [x] Gizli katmanı ve embedding boyutunu büyüt, embedding'leri 2 boyutta çizdir, modelden isim örnekle
- [x] Part 3: başlangıç loss'unun neden yüksek olduğunu ve tanh'ın neden doyduğunu göster, Kaiming init ile düzelt
- [x] BatchNorm katmanını ekle: eğitimde batch istatistiği, tahminde running mean
- [ ] Türkçe isim listesiyle aynı modeli eğit, örnekler + dev loss, bigram'ın Türkçe sonuçlarıyla karşılaştır
- [ ] *(opsiyonel)* Karpathy'nin ek egzersizleri (sıfır init teşhisi, BatchNorm'u Linear'a katlama, 2.2 val loss'u geçme)

## Bu klasördeki dosyalar
- `mlp.py` — Part 2: veri seti, embedding, MLP, minibatch eğitimi, lr taraması, train/dev/test, embedding çizimi, sampling
- `mlp-2.py` — Part 3: init düzeltmeleri, Kaiming init, elle yazılmış BatchNorm, sonra `Linear`/`BatchNorm1d`/`Tanh` sınıflarıyla 6 katmanlı ağ ve 4 teşhis grafiği
- `names.txt` — Karpathy'nin İngilizce isim listesi
- `output/` — loss eğrisi, aktivasyon/gradyan dağılımları, ağırlık-gradyan dağılımı, update-ratio grafiği

## Ölçümler

Part 3'teki iyileştirmelerin dev loss'a etkisi (`mlp-2.py` içinde yorum olarak da duruyor):

| Aşama | train | val |
|---|---|---|
| başlangıç | 2.1245 | 2.1682 |
| softmax'ın aşırı güvenini düzelt | 2.07 | 2.13 |
| tanh'ın init'te doymasını düzelt | 2.0356 | 2.1027 |
| Kaiming init | 2.0377 | 2.1070 |
| BatchNorm eklendi | 2.0668 | 2.1048 |

**Okunuşu:** asıl sıçrama ilk iki init düzeltmesinden geliyor (2.1682 → 2.1027). Kaiming ve BatchNorm bu ölçekte dev loss'u kayda değer biçimde iyileştirmiyor — değerleri daha derin ağlarda ve eğitimi dayanıklı kılmakta ortaya çıkıyor. Part 3'ün asıl kazancı sayı değil, teşhis araçları: aktivasyon/gradyan histogramları ve update-ratio grafiği.

## Durum

Görev 1-6 tamamlandı, 13 Eylül'de teslim edildi (repo linki + video maile cevap olarak gönderildi).

**Açık kalan:** Görev 7 (Türkçe veri — Hafta 3 Görev 5'ten beri açık) ve Görev 8 (opsiyonel). Hafta 5 maili eksiklerin bu klasöre sonradan eklenmesine açıkça izin veriyor.

**Not (2026-09-18):** Kod çalışıyor ama tek günde videoyla birlikte yazıldığı için içselleşmedi. Forward pass'i ara değişkenlere bölerek yeniden yazma kararı alındı — o yeniden yazım Hafta 5'in birinci adımı, çünkü Hafta 5 tam olarak bu ara değişkenlerin gradient'ini elle istiyor.
