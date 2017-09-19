import numpy as np
from sklearn import linear_model
from sklearn import datasets
import matplotlib.pyplot as plt 

# Create this numpy array:
# array([[ 0.5],
#        [ 1. ]])
X = np.c_[.5, 1].T

y = [.5, 1]

# Create this array:
# array([[0],
#        [2]])
test = np.c_[0,2].T

regr = linear_model.LinearRegression()

plt.figure()

np.random.seed(0)

# Create 6 new data points based on X with a small amount of normally distributed random noise added
for _ in range(6):
	this_X = .1*np.random.normal(size=(2,1)) + X
	regr.fit(this_X, y)
	plt.plot(test, regr.predict(test))

plt.show()

# Now do a ridge regression
regr = linear_model.Ridge(alpha=.1)

plt.figure()

np.random.seed(0)
for _ in range(6):
	this_X = .1*np.random.normal(size=(2,1)) + X
	regr.fit(this_X, y)
	plt.plot(test, regr.predict(test))
	plt.scatter(this_X, y, s=3)

plt.show()

