from datasets import load_dataset
import numpy as np



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

def classify(X, w):
    return np.round(forward(X, w))


def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)

def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (forward(X,w) - Y)) / X.shape[0]


def train_model(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        if (i % 10 == 0):
            print(f'Iterations {i:4d} => Loss: {loss(X, Y, w):.20f}')
        w -= gradient(X, Y, w) * lr

    return w


def test_model(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" %
            (correct_results, total_examples, success_percent))

# veriyi modelin istedigi sekle sokma

def prepare_X(images):
    """(m, 28, 28) uint8 goruntuler -> (m, 785) matris.

    Her goruntu tek satira duzlestiriliyor (28*28 = 784), sonra basa
    hep 1 olan bias sutunu ekleniyor, boylece w[0] bias oluyor.
    """
    flat = images.reshape(images.shape[0], -1)              # (m, 784)
    return np.column_stack((np.ones(flat.shape[0]), flat))  # (m, 785)


def encode_labels(labels, digit):
    return (labels == digit).astype(int).reshape(-1, 1)


DIGIT = 5  # hangi rakami taniyacagiz

ds = load_dataset("ylecun/mnist")
train = ds["train"].with_format("numpy")
test = ds["test"].with_format("numpy")

X_train = prepare_X(np.asarray(train["image"]))
Y_train = encode_labels(np.asarray(train["label"]), DIGIT)
X_test = prepare_X(np.asarray(test["image"]))
Y_test = encode_labels(np.asarray(test["label"]), DIGIT)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape,
      "- icinde", int(Y_train.sum()), "tane", DIGIT)
print("X_test :", X_test.shape, "Y_test :", Y_test.shape,
      "- icinde", int(Y_test.sum()), "tane", DIGIT)

w = train_model(X_train, Y_train, iterations=100, lr=1e-5)
test_model(X_test, Y_test, w)
