# Hafta 2 — Backpropagation

**Teslim tarihi:** 29 Ağustos 2026

## Bu hafta öğrenilecekler
- Chain rule (zincir kuralı) sezgisi
- Computation graph: her işlem bir node, gradient çıktıdan girdiye doğru akar
- Backward pass: her node kendi local derivative'ini üstten gelen gradient ile çarpar
- Analitik gradient'in numerical derivative ile doğrulanması (Hafta 1'in devamı)
- PyTorch'un autograd'ının çekirdeğinde de aynı mekanizma olduğu

## Kaynaklar
- [Andrej Karpathy — The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) (videonun tamamı, 2.5 saat)
- [Karpathy'nin micrograd reposu](https://github.com/karpathy/micrograd) (referans)
- [3Blue1Brown — What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [3Blue1Brown — Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)

## Görevler
- [x] Kendi `Value` sınıfını yaz — toplama ve çarpma, computation graph
- [x] Gradient'leri elle doldur: basit bir ifade, sonra `tanh` içeren bir nöron
- [x] `backward()` metodunu yaz: ters topolojik sıra, çoklu-yol durumunda gradient toplama
- [ ] `tanh`'ı parçala (`exp`, bölme, `pow`), `backward()`/numerical derivative/PyTorch ile üçlü doğrula
- [x] `Neuron`, `Layer`, `MLP` sınıflarını kur, küçük bir veri setiyle eğit

**Durum:** Görev 1-3 tamamlandı. Görev 4 kısmi — `tanh` ve `sigmoid`, `exp`/`pow`/bölme primitifleriyle parçalandı ve `backward()` ile doğru gradyan verdiği doğrulandı; 2026-09-06'da PyTorch karşılaştırması da eklendi (`torch_test()`) ve değerler tuttu, ama sayısal türev ayağı ile üçünü yan yana basan tolerans kontrolü hâlâ eksik. Görev 5 2026-09-06'da tamamlandı: `Neuron`/`Layer`/`MLP` + `parameters()` yazıldı, 4 örneklik veri setiyle 30 adım eğitildi, loss 2.1125 → 0.0169. Deadline'da (29 Ağustos) elde olanla teslim edilmişti; bu maddeler sonradan kapatıldı.
