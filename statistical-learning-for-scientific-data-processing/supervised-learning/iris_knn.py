import numpy as np 
from sklearn import datasets

iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

# Find the unique values in the target values
print()
print("The unique values in the target array are: ", np.unique(iris_y))

# Seed the PRNG
np.random.seed(0)

# Create an randomized indicies array of the same length as the sample array
indicies = np.random.permutation(len(iris_X))

# Set aside the first 10 data points as test data
iris_X_train = iris_X[indicies[:-10]]
iris_y_train = iris_y[indicies[:-10]]
iris_X_test = iris_X[indicies[-10:]]
iris_y_test = iris_y[indicies[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
print("Creating a new instance of a KNN classifier.")
knn = KNeighborsClassifier()
print("Fitting the model:", knn.fit(iris_X_train, iris_y_train))

# Predict values in the test set
knn.predict(iris_X_test)

print("The predicted values are:", iris_y_test)
