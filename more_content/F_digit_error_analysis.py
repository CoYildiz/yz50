"""Hangi rakam neden zor? Ikili siniflandiricinin hatalarini ayristirir.

Her rakam icin "bu rakam mi" diye soran ayri bir ikili siniflandirici egitiliyor,
sonra hatalari iki gruba ayriliyor:

  false negative (FN) : gercek rakam BU, model "hayir" dedi      -> kacirdi
  false positive (FP) : gercek rakam BASKA, model "evet" dedi    -> karistirdi

FP'ler ayrica gercek etiketlerine gore dagitiliyor, boylece "8'i en cok neyle
karistiriyor" sorusu cevaplaniyor.
"""

import numpy as np
from datasets import load_dataset


# --- model (D ve E ile ayni) ---------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w):
    return sigmoid(np.matmul(X, w))


def classify(X, w):
    return np.round(forward(X, w))


def loss(X, Y, w):
    y_hat = np.clip(forward(X, w), 1e-15, 1 - 1e-15)
    return -np.average(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))


def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for _ in range(iterations):
        w -= gradient(X, Y, w) * lr
    return w


# --- veri -----------------------------------------------------------------

def prepare_X(images):
    flat = images.reshape(images.shape[0], -1)
    return np.column_stack((np.ones(flat.shape[0]), flat))


def load_mnist():
    ds = load_dataset("ylecun/mnist")
    tr = ds["train"].with_format("numpy")
    te = ds["test"].with_format("numpy")
    return (prepare_X(np.asarray(tr["image"])), np.asarray(tr["label"]),
            prepare_X(np.asarray(te["image"])), np.asarray(te["label"]))


# --- analiz ---------------------------------------------------------------

def analyse(digit, X_train, y_train, X_test, y_test, iterations, lr):
    """Tek bir rakam icin egit, test et, hatalari ayristir."""
    Y_train = (y_train == digit).astype(int).reshape(-1, 1)
    w = train(X_train, Y_train, iterations, lr)

    pred = classify(X_test, w).astype(int).ravel()   # 0/1
    truth = (y_test == digit).astype(int)            # 0/1

    tp = int(np.sum((pred == 1) & (truth == 1)))
    tn = int(np.sum((pred == 0) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))     # kacirdiklari
    fp = int(np.sum((pred == 1) & (truth == 0)))     # karistirdiklari

    # FP'leri gercek etiketlerine gore dagit: hangi rakamlari bu sanmis
    fp_by_digit = np.bincount(y_test[(pred == 1) & (truth == 0)], minlength=10)

    n_positive = int(truth.sum())
    return {
        "digit": digit, "w": w,
        "tp": tp, "tn": tn, "fn": fn, "fp": fp,
        "accuracy": (tp + tn) / len(y_test) * 100,
        "baseline": (len(y_test) - n_positive) / len(y_test) * 100,
        "recall": tp / n_positive * 100,                       # kacini yakaladi
        "precision": tp / (tp + fp) * 100 if tp + fp else 0.0,  # "evet"lerinin kaci dogru
        "fp_by_digit": fp_by_digit,
        "n_positive": n_positive,
    }


def main(iterations=100, lr=1e-5):
    X_train, y_train, X_test, y_test = load_mnist()

    results = []
    for d in range(10):
        r = analyse(d, X_train, y_train, X_test, y_test, iterations, lr)
        results.append(r)
        print(f"rakam {d} bitti", end="\r")

    print(" " * 20, end="\r")
    print("rakam  adet  hep-hayir  accuracy  kazanc |   FN    FP | recall precision")
    print("-" * 76)
    for r in results:
        print(f"{r['digit']:>5} {r['n_positive']:>5} "
              f"{r['baseline']:>9.2f}% {r['accuracy']:>8.2f}% "
              f"{r['accuracy'] - r['baseline']:>+6.2f} |"
              f"{r['fn']:>5}{r['fp']:>6} | "
              f"{r['recall']:>5.1f}% {r['precision']:>8.1f}%")

    # En zor rakam icin: neyi bu sandi?
    hardest = min(results, key=lambda r: r["accuracy"] - r["baseline"])
    d = hardest["digit"]
    print(f"\nEn zor rakam: {d}")
    print(f"  kacirdigi (FN)     : {hardest['fn']:>4}  / {hardest['n_positive']} gercek {d}")
    print(f"  yanlis evet (FP)   : {hardest['fp']:>4}")
    print(f"\n  '{d}' sandigi rakamlar:")
    order = np.argsort(hardest["fp_by_digit"])[::-1]
    for other in order:
        n = int(hardest["fp_by_digit"][other])
        if n == 0:
            continue
        print(f"    gercek {other}: {n:>4} kez {d} sanildi")

    return results


if __name__ == "__main__":
    main()
