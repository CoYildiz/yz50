import time

import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt



sea.set()
plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)
plt.xticks(fontsize=14)                                  # set x axis ticks
plt.yticks(fontsize=14)                                  # set y axis ticks
plt.xlabel("Reservations", fontsize=14)                  # set x axis label
plt.ylabel("Pizzas", fontsize=14)                        # set y axis label
X, Y = np.loadtxt("data/pizza.txt", skiprows=1, unpack=True)  # load data
plt.plot(X, Y, "bo")                                     # plot data
plt.savefig("output/basic_lin_reg/pizza.png") # we saved already so it is not neccesary for now

# Calling the predict() function
def predict(X, w, b):
    return X * w + b
# Calculatin the loss
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

# computing the derivative
def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)

# calling the training function for 20,000 iterations
def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        if (i % 5000 == 0):
            print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b

w, b = train(X, Y, iterations=10000, lr=0.001)
print(f"w:{w}, b:{b}")


plt.plot(X, Y, "bo")
plt.xlabel("Reservations")
plt.ylabel("Pizzas")
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
plt.savefig("output/basic_lin_reg/withline.png")

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))



#---------
# closed form (least squares), y = a + bx
# weight: b = mean(X*Y) - mean(X)*mean(Y)         Cov(X, Y)
#             ----------------------------    =  ---------
#             mean(X**2) - pow(mean(X), 2)        Var(X)
# bias:   a = mean(Y) - b * mean(X)
#
# same thing written with sums:
#         b = n*sum(X*Y) - sum(X)*sum(Y)
#             -----------------------------
#             n*sum(X**2) - pow(sum(X), 2)
# the n factors belong to BOTH terms (sum form) or to NEITHER (mean form) -
# mixing the two was the bug here: it inflated the slope to 1.84 instead of 1.08.
#
# sum error might be 1 / (n-2)  -- residual variance, two parameters fitted; slope unaffected
n = len(X)
def least_squares(X, Y):
    weight = (np.mean(X*Y) - np.mean(X)*np.mean(Y)) / (np.mean(X**2) - np.pow(np.mean(X), 2))
    bias = np.mean(Y) - weight * np.mean(X)
    return weight, bias
weight, bias = least_squares(X, Y)
print(f"weight: {weight}, bias: {bias}")
print(20*weight + bias)
