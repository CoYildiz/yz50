# Hafta 3 — Dil Modelleme (Bigram, makemore)

**Teslim tarihi:** 6 Eylül 2026

## Bu hafta öğrenilecekler
- Bigram karakter modeli: bir harften sıradaki harfin olasılığı
- Sayım tablosunu olasılık dağılımına çevirme ve modelden sampling
- Negative log likelihood: modelin kalitesini tek sayıyla ölçme, neden bu loss
- One-hot encoding, logits ve softmax ile olasılığa geçiş
- Aynı bigram modelinin hem sayımla hem gradient descent ile kurulabildiği
- Hafta 2'de yazılan `backward()`'ın burada da işin çekirdeği olduğu — PyTorch aynı mekanizmayı tensor'larla çalıştırıyor

## Kaynaklar
- [Andrej Karpathy — The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) (~2 saat)
- [Karpathy'nin makemore reposu](https://github.com/karpathy/makemore) — referans, `names.txt` de burada
- [PyTorch broadcasting kuralları](https://pytorch.org/docs/stable/notes/broadcasting.html) — Görev 2'deki `keepdim` için

## Görevler
- [x] Bigram'ları önce Python dictionary, sonra 27x27 torch tensor'da say; tabloyu görselleştir
- [x] Sayım tablosunu satır satır olasılığa çevir (`keepdim=True`), modelden yeni isimler örnekle
- [x] Negative log likelihood'u hesapla *(smoothing henüz eklenmedi)*
- [x] Aynı modeli tek katmanlı sinir ağıyla kur: one-hot, 27x27 weight, softmax, NLL, gradient descent
- [ ] Türkçe isim listesi bul/temizle, alfabeyi ç/ğ/ı/ö/ş/ü ile genişlet, iki modeli de koştur
- [ ] *(opsiyonel)* Trigram'a çevir, train/dev/test böl, bigram ile karşılaştır

## Bu klasördeki dosyalar
- `makemore.py` — asıl çalışma: sayım, olasılık, sampling, NLL
- `names.txt` — Karpathy'nin İngilizce isim listesi (32.033 isim)
- `animate_sampling.py` — sampling'in 27x27 matris üzerindeki yürüyüşü (animasyon)
- `code_sampling_animation.py` — aynı döngü, kod satırı satırı: hangi satır çalıştı, ne değişti
- `nll_animation.py` — NLL'in bigram bigram birikmesi + sıfır olasılığın neden smoothing gerektirdiği
- `output/` — üretilen grafikler ve animasyonlar

## Doğrulama (2026-09-06)
- `N.sum()` = 228.146 (toplam bigram)
- `N[0].sum()` = 32.033 = kelime sayısı — her kelime tam bir başlangıç bigram'ı üretir
- `P.sum(1)` hepsi 1, `P.sum(0)` değil — satır bazında normalize edildiğinin kanıtı
- Tüm veri kümesi NLL = 2.4540

**Durum (2026-09-20 güncellendi):** Görev 1-4 tamamlandı. Görev 4, 13 Eylül'de yazıldı
(`8af6389`): `xs`/`ys` eğitim seti, `F.one_hot` ile 27x27 girdi, `W` üzerinden logits, satır bazında
normalize ile softmax, ortalama NLL ve gradient descent döngüsü — loss 2.50 civarına indi, sayım
modelinin 2.4540'ına yaklaştı. Bu, iki modelin aynı şeyi öğrendiğinin kanıtı.

**Açık kalanlar:**
- Görev 3'ün smoothing kısmı hiç eklenmedi (`N + 1` yerine farklı sahte sayımların loss'a etkisi)
- Görev 5 (Türkçe isim listesi) **yapılmadı** — `makemore.py` yalnızca İngilizce `names.txt` ile
  çalışıyor, alfabe Türkçe karakterlerle genişletilmedi. Bu madde Hafta 4 Görev 7 olarak da açık
- Görev 6 opsiyoneldi, açılmadı
