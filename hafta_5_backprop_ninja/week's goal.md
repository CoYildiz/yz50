# Hafta 5 — Backprop Ninja (elle gradient)

**Teslim tarihi:** 20 Eylül 2026 Pazar, 19:00

## Bu hafta öğrenilecekler
- Her ara değişkenin gradient'ini elle yazmak ve PyTorch'unkiyle karşılaştırmak
- Broadcasting'in gradient'e etkisi: toplanan boyut geri dönerken nerede `sum` alınır
- Cross entropy ve BatchNorm'un türevlerinin neden tek satıra indiği

## Kaynaklar
- [Karpathy — Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) (~1s55d)
- [Egzersiz notebook'u](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore) — `makemore_part4_backprop.ipynb`
- [cs231n — Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — matris düzeyinde backprop
- [Vector, Matrix, and Tensor Derivatives](https://cs231n.stanford.edu/handouts/derivatives.pdf) — broadcasting'in geri dönüşünde `sum`'ın nereden çıktığı

## Görevler
- [ ] Hafta 4'ün MLP + BatchNorm modelini ara değişkenlere böl ve `loss.backward()` ile her ara değişkenin gradient'ini al
- [ ] Aynı gradient'leri elle yaz, `cmp` fonksiyonuyla tek tek karşılaştır — hepsi "exact" olana kadar
- [ ] *(opsiyonel)* Cross entropy ve BatchNorm'un geriye yayılımını tek ifadeye indir, modeli `loss.backward()` olmadan eğit

## Ön koşul: forward pass ara değişkenlere bölünmüş olmalı

Hafta 4'te forward tek satıra sıkıştırılmıştı (`F.cross_entropy(logits, Y)`). Bu hafta o zincirin
her halkasının gradient'i ayrı yazılacağı için forward'ın açılması gerekiyor:

```
logit_maxes → norm_logits → counts → counts_sum → counts_sum_inv → probs → logprobs → loss
```

BatchNorm de `std` ile değil varyans zinciriyle:

```
bnmeani → bndiff → bndiff2 → bnvar → bnvar_inv → bnraw → hpreact
```

**Neden `counts_sum_inv` ayrı bir değişken:** `probs = counts * counts_sum_inv` yazılırsa türev
çarpım kuralına düşüyor; `counts / counts_sum` yazılırsa aynı iş bölmenin türeviyle yapılıyor.

**Not:** Bu hafta eğitilmiş model gerekmiyor. Tek bir minibatch (32 örnek) ve rastgele init yeterli —
egzersiz gradient'lerin doğruluğunu ölçüyor, modelin kalitesini değil.

## Teslim checklist
- [ ] Kod bu klasöre eklendi ("yz50" collaborator ekli kalmalı)
- [ ] Video: **en çok zorlanılan türev** ve nasıl doğrulandığı anlatıldı
- [ ] Repo linki + video, göreve gelen maile cevap olarak gönderildi

## Durum

Başlanmadı.
