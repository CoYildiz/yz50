"""
makemore part 3: activations & gradients, BatchNorm

Iki bolum var:
  1) Elle yazilmis MLP + BatchNorm  (init duzeltmeleri: softmax'i az guvenli yap,
     tanh'i doyurma, kaiming init, sonra batchnorm)
  2) "PyTorchifying": Linear / BatchNorm1d / Tanh siniflari ile 6 katmanli ag +
     aktivasyon, gradyan ve update:data orani teshis grafikleri

Calistirma: bu klasorden `python mlp-2.py`. Grafikler output/ altina kaydedilir.
"""

import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# DEBUG=True  -> 2. bolumde 1000 adim kosar, teshis grafiklerini uretir (videodaki hali)
# DEBUG=False -> tam 200k adim egitim, sonra sampling
DEBUG = True
# BONUS=True  -> videoda anlatilmayan ek istatistik deneyleri (en altta)
BONUS = False

# -----------------------------------------------------------------------------
# veri
# -----------------------------------------------------------------------------

words = open('names.txt', 'r').read().splitlines()
print(words[:8])
print(len(words))

# karakter -> integer sozlukleri
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

block_size = 3  # context length: kac karaktere bakip bir sonrakini tahmin ediyoruz


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr,  Ytr = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])  # 10%
Xte,  Yte = build_dataset(words[n2:])     # 10%

# -----------------------------------------------------------------------------
# 1) MLP revisited: elle yazilmis BatchNorm
# -----------------------------------------------------------------------------

n_embd = 10    # karakter embedding boyutu
n_hidden = 200  # gizli katmandaki noron sayisi

g = torch.Generator().manual_seed(2147483647)  # tekrar uretilebilirlik icin
C = torch.randn((vocab_size, n_embd),             generator=g)
# kaiming init: gain / sqrt(fan_in). tanh icin gain = 5/3
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / ((n_embd * block_size) ** 0.5)
# b1'e gerek yok: BatchNorm ortalamayi cikardigi icin bias iptal oluyor, isini bnbias goruyor
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01  # cikis katmani: baslangicta az "guvenli"
b2 = torch.randn(vocab_size,                      generator=g) * 0

# BatchNorm parametreleri
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))  # toplam parametre sayisi
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb]                          # karakterleri vektore gom
    embcat = emb.view(emb.shape[0], -1)  # vektorleri birlestir
    # Linear katman
    hpreact = embcat @ W1
    # BatchNorm katmani ------------------------------------------------------
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    with torch.no_grad():
        # egitim sirasinda kosan (running) ortalama/std tut ki sonra tek ornek
        # icin de tahmin yapabilelim
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    # ------------------------------------------------------------------------
    h = torch.tanh(hpreact)   # non-linearity
    logits = h @ W2 + b2      # cikis katmani
    loss = F.cross_entropy(logits, Yb)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01  # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

plt.figure(figsize=(10, 4))
plt.plot(lossi)
plt.title('loss (log10) - elle yazilmis BatchNorm')
plt.savefig('output/mlp2_loss.png')
plt.close()

# egitim bittikten sonra batchnorm'u kalibre etmenin alternatif yolu:
# tum egitim setini gecirip ortalama/std'yi bir kerede olcmek.
# (running estimate zaten tuttugumuz icin buna gerek yok, karsilastirma icin duruyor)
with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1
    bnmean = hpreact.mean(0, keepdim=True)
    bnstd = hpreact.std(0, keepdim=True)


@torch.no_grad()  # bu decorator gradyan takibini kapatir
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x]                           # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1)  # (N, block_size * n_embd)
    hpreact = embcat @ W1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)              # (N, n_hidden)
    logits = h @ W2 + b2                 # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


split_loss('train')
split_loss('val')

# loss log
# ---------
# original:                                       train 2.1245 / val 2.1682
# fix softmax confidently wrong:                  train 2.07   / val 2.13
# fix tanh layer too saturated at init:           train 2.0356 / val 2.1027
# use semi-principled "kaiming init":             train 2.0377 / val 2.1070
# add batch norm layer:                           train 2.0668 / val 2.1048

# -----------------------------------------------------------------------------
# 2) SUMMARY + PYTORCHIFYING: derin bir ag kuralim
#    Asagidaki siniflar PyTorch'taki nn.Module API'siyle ayni arayuzu tasiyor
# -----------------------------------------------------------------------------


class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parametreler (backprop ile ogrenilir)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffer'lar ('momentum update' ile guncellenir, gradyani yok)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)  # batch mean
            xvar = x.var(0, keepdim=True)    # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # birim varyansa normalize et
        self.out = self.gamma * xhat + self.beta
        # buffer guncelle
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


n_embd = 10    # karakter embedding boyutu
n_hidden = 100  # gizli katmandaki noron sayisi
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size, n_embd), generator=g)
layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]
# BatchNorm'suz hali (karsilastirma icin, gain 5/3 gerekiyor):
# layers = [
#   Linear(n_embd * block_size, n_hidden), Tanh(),
#   ... ,
#   Linear(           n_hidden, vocab_size),
# ]

with torch.no_grad():
    # son katman: baslangicta daha az "guvenli" olsun
    layers[-1].gamma *= 0.1
    # diger katmanlar: gain uygula (BatchNorm varken 1.0 yetiyor, yoksa 5/3)
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 1.0  # 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))  # toplam parametre sayisi
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []
ud = []  # update:data orani (log10)

for i in range(max_steps):

    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb]                     # karakterleri vektore gom
    x = emb.view(emb.shape[0], -1)  # vektorleri birlestir
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    if DEBUG:
        for layer in layers:
            layer.out.retain_grad()  # ara aktivasyonlarin gradyanina bakabilmek icin
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 150000 else 0.01  # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in parameters])

    if DEBUG and i >= 1000:
        break  # teshis grafikleri icin ilk 1000 adim yeterli

# --- teshis 1: aktivasyon dagilimi (tanh ciktilari) ---------------------------
if DEBUG:
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):  # cikis katmanini haric tut
        if isinstance(layer, Tanh):
            t = layer.out
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%'
                  % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title('activation distribution')
    plt.savefig('output/mlp2_activation_dist.png')
    plt.close()

    # --- teshis 2: gradyan dagilimi (tanh ciktilarinin gradyanlari) -----------
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = layer.out.grad
            print('layer %d (%10s): mean %+f, std %e'
                  % (i, layer.__class__.__name__, t.mean(), t.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title('gradient distribution')
    plt.savefig('output/mlp2_gradient_dist.png')
    plt.close()

    # --- teshis 3: agirlik gradyanlari ---------------------------------------
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):
        t = p.grad
        if p.ndim == 2:
            print('weight %10s | mean %+f | std %e | grad:data ratio %e'
                  % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} {tuple(p.shape)}')
    plt.legend(legends)
    plt.title('weights gradient distribution')
    plt.savefig('output/mlp2_weight_grad_dist.png')
    plt.close()

# --- teshis 4: update:data orani (hedef ~1e-3, yani grafikte -3) --------------
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k')  # bu oranlar ~1e-3 olmali
plt.legend(legends)
plt.title('update:data ratio (log10)')
plt.savefig('output/mlp2_update_ratio.png')
plt.close()

# -----------------------------------------------------------------------------
# degerlendirme + sampling
# -----------------------------------------------------------------------------

# katmanlari eval moduna al (BatchNorm artik running estimate kullanacak)
for layer in layers:
    layer.training = False


@torch.no_grad()
def split_loss_deep(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x]                      # (N, block_size, n_embd)
    x = emb.view(emb.shape[0], -1)  # (N, block_size * n_embd)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)
    print(split, loss.item())


split_loss_deep('train')
split_loss_deep('val')

# modelden isim uret
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size  # ... ile basla
    while True:
        # forward pass
        emb = C[torch.tensor([context])]  # (1, block_size, n_embd)
        x = emb.view(emb.shape[0], -1)
        for layer in layers:
            x = layer(x)
        logits = x
        probs = F.softmax(logits, dim=1)
        # dagilimdan ornekle
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        # context penceresini kaydir
        context = context[1:] + [ix]
        out.append(ix)
        # ozel '.' tokenini cekersek dur
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

# -----------------------------------------------------------------------------
# BONUS (videoda anlatilmadi): forward/backward pass'te aktivasyon istatistikleri
# -----------------------------------------------------------------------------

if BONUS:
    # sadece Linear: cikisin std'si patliyor (31x), backward'da da simetrik olarak
    # a'nin gradyani patliyor
    g = torch.Generator().manual_seed(2147483647)

    a = torch.randn((1000, 1), requires_grad=True, generator=g)     # a.grad = b.T @ c.grad
    b = torch.randn((1000, 1000), requires_grad=True, generator=g)  # b.grad = c.grad @ a.T
    c = b @ a
    loss = torch.randn(1000, generator=g) @ c
    a.retain_grad()
    b.retain_grad()
    c.retain_grad()
    loss.backward()
    print('a std:', a.std().item())
    print('b std:', b.std().item())
    print('c std:', c.std().item())
    print('-----')
    print('c grad std:', c.grad.std().item())
    print('a grad std:', a.grad.std().item())
    print('b grad std:', b.grad.std().item())

    # Linear + BatchNorm: BatchNorm cikisi birim varyansa cekiyor, gradyanlar da
    # olcekten bagimsiz hale geliyor
    g = torch.Generator().manual_seed(2147483647)

    n = 1000
    # linear katman ---
    inp = torch.randn(n, requires_grad=True, generator=g)
    w = torch.randn((n, n), requires_grad=True, generator=g)  # / n**0.5
    x = w @ inp
    # bn katmani ---
    xmean = x.mean()
    xvar = x.var()
    out = (x - xmean) / torch.sqrt(xvar + 1e-5)
    # ---
    loss = out @ torch.randn(n, generator=g)
    inp.retain_grad()
    x.retain_grad()
    w.retain_grad()
    out.retain_grad()
    loss.backward()

    print('inp std: ', inp.std().item())
    print('w std: ', w.std().item())
    print('x std: ', x.std().item())
    print('out std: ', out.std().item())
    print('------')
    print('out grad std: ', out.grad.std().item())
    print('x grad std: ', x.grad.std().item())
    print('w grad std: ', w.grad.std().item())
    print('inp grad std: ', inp.grad.std().item())
