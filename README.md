# yz50-todo

**YZ50** — Türkiye'den 50 kişinin seçildiği, 12 haftalık bir yapay zeka araştırmacılığı programı — kapsamında yaptığım haftalık ödevlerin deposu. Program, Andrej Karpathy'nin "Zero to Hero" yaklaşımını izliyor: elle backpropagation, sıfırdan bir dil modeli, transformer, mini-GPT.

## İçerik

| Hafta | Konu | Durum | Detay |
|---|---|---|---|
| [hafta_1](hafta_1/) | Neural Networks — neuron, forward pass, loss, gradient descent (sayısal türev) | Tamamlandı | [week's goal.md](hafta_1/week's%20goal.md) |
| [hafta_2_backpropagation](hafta_2_backpropagation/) | Backpropagation — `Value` sınıfı, computation graph, `backward()`, `tanh`/`exp`/`pow` parçalama | Kısmi | [week's goal.md](hafta_2_backpropagation/week's%20goal.md) |

Her haftanın klasöründeki `week's goal.md`, o haftanın öğrenme hedeflerini, kaynaklarını ve görev listesini içerir.

## Kurulum

```bash
uv sync
```

Bağımlılıklar: `numpy`, `matplotlib`, `graphviz` (computation graph görselleştirme), `torch` (gradyan doğrulama için referans).
