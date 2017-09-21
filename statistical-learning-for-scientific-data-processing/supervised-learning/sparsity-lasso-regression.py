# This program demonstrates the effect of changing the alpha on the model score.
# Alpha is the regularization strength.

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

# now try Lasso Regression
print()
print("Lasso regression:")
print("-----------------")

alphas = np.logspace(-4, -1, 6)
print("alphas:", [alph for alph in alphas])

regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha
	).fit(diabetes_X_train, diabetes_y_train
	).score(diabetes_X_test, diabetes_y_test)
	for alpha in alphas]
print("scores:", scores)

print("Regression coefficients:", regr.coef_)