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
plt.show()
