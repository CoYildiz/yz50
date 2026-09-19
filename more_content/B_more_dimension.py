# Plot a plane that roughly approximates a dataset with two input variables.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import seaborn as sns

# Import the dataset
x1, x2, x3, y = np.loadtxt("data/pizza_3_vars.txt", skiprows=1, unpack=True)
# The first column is all ones: it is a fake input whose value is always 1,
# so w[0] is multiplied by 1 for every example -> w[0] IS the bias.
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1,1)

def predict(X, w):
    return np.matmul(X, w)

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)/ X.shape[0])


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

w = train(X, Y, iterations=50000, lr=0.001)
print("bias (w[0]): %.4f" % w[0])
print("weights     :", w[1:].ravel())

# Plot AFTER training, so the plane uses the learned weights.
# x1 = Reservations, x2 = Temperature (see the column order in pizza_3_vars.txt).
sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("Reservations", labelpad=15, fontsize=15)
ax.set_ylabel("Temperature", labelpad=15, fontsize=15)
ax.set_zlabel("Pizzas", labelpad=5, fontsize=15)

# Plot the data points
ax.scatter3D(x1, x2, y, color='b')

# Plot the plane
MARGIN = 10
edges_x = [np.min(x1) - MARGIN, np.max(x1) + MARGIN]
edges_y = [np.min(x2) - MARGIN, np.max(x2) + MARGIN]
xs, ys = np.meshgrid(edges_x, edges_y)
zs = np.array([w[0] + x * w[1] + y * w[2] for x, y in
              zip(np.ravel(xs), np.ravel(ys))])
ax.plot_surface(xs, ys, zs.reshape((2, 2)), alpha=0.2)

plt.savefig("outputs/more_dimensions/pizza_3_vars.png")
