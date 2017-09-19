import numpy as np 
from sklearn import datasets

iris = datasets.load_iris()

iris_X = iris.data 
iris_y = iris.target

# Set aside the first 10 data points as test data
indicies = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indicies[:-10]]
iris_y_train = iris_y[indicies[:-10]]
iris_X_test = iris_X[indicies[-10:]]
iris_y_test = iris_y[indicies[-10:]]

from sklearn import linear_model

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train, iris_y_train)

print("Coefficients:", logistic.coef_)

# Predict the test set 
print("Predicted test values:", logistic.predict(iris_X_test))
print("Actual test values:   ", iris_y_test)

