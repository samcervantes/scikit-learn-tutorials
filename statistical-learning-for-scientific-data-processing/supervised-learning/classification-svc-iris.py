import numpy as np 
from sklearn import datasets, svm

iris = datasets.load_iris()

iris_X = iris.data 
iris_y = iris.target

# Set aside the first 10 data points as test data
indicies = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indicies[:-10]]
iris_y_train = iris_y[indicies[:-10]]
iris_X_test = iris_X[indicies[-10:]]
iris_y_test = iris_y[indicies[-10:]]

svc = svm.SVC(kernel='linear')

print(svc.fit(iris_X_train, iris_y_train))