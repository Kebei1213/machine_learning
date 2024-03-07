import matplotlib.pyplot as plt
import numpy as np

import iris

plt.rcParams['font.family'] = 'Arial Unicode MS'

w1, w2 = iris.model.w.reshape(2, -1)
x1 = iris.X[iris.y == 1, :]
plt.scatter(x1[:, 0], x1[:, 1], label="山鸢尾花")
x2 = iris.X[iris.y == -1, :]
plt.scatter(x2[:, 0], x2[:, 1], label="非山鸢尾花")
lx = np.array([4.2, 7.5])
ly = (-w1 * lx - iris.model.b) / w2
plt.plot(lx, ly)
plt.legend(loc='upper right')
plt.show()
