import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

iris = datasets.load_iris()
# print(iris)
X = iris["data"][:, (0, 1)]
#所有行，0，1列
y = 2 * (iris["target"] == 0).astype(np.int32) - 1
#iris["target"] == 0若为0则是Ture or False
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
#test_size=0.4  表示40为测试，60为训练

model = Perceptron()
model.fit(X_train, y_train)
model.predict(X_test)