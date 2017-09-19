from __future__ import print_function
from sklearn import datasets
import numpy as np 

diabetes = datasets.load_diabetes()

# Set aside the last 20 samples for test data
diabetes_X_test = diabetes.data[-20:]
diabetes_y_test = diabetes.target[-20:]

diabetes_X_train = diabetes.data[:-20]
diabetes_y_train = diabetes.target[:-20]

from sklearn import linear_model

# Create a new instance of a linear regression model
regr = linear_model.LinearRegression()

# Train the model
regr.fit(diabetes_X_train, diabetes_y_train)

print()
print("Linear coefficients:", regr.coef_)
print()
print("X_test:")
print("-------")
print(diabetes_X_test)
print()
print("y_test:")
print("-------")
print(diabetes_y_test)
print()

predicted_values = regr.predict(diabetes_X_test)
print("Predicted Test Values:", predicted_values)
print()

# Calculate the mean squared error
mse = np.mean((predicted_values-diabetes_y_test)**2)
print("Mean squared errror:", mse)

# now try Ridge Regression
print()
print("Ridge regression:")
print("-----------------")

regr = linear_model.Ridge(alpha=.1)

# Generate an array of six evenly spaced numbers on a log scale from 10^-4 to 10^-1:
# array([ 0.0001,  0.00039811, 0.00158489, 0.00630957, 0.02511886, 0.1])
alphas = np.logspace(-4, -1, 6)
print("alphas:", [alph for alph in alphas])

print("scores:", [regr.set_params(alpha=alph).fit(diabetes_X_train, diabetes_y_train,).score(diabetes_X_test, diabetes_y_test) for alph in alphas])


