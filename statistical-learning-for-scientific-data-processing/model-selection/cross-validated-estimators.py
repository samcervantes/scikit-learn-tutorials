# This program uses the LassoCV model which is a linear regression model that chooses it's own 
#   lambda based on cross-validation

from sklearn import linear_model, datasets
lasso = linear_model.LassoCV()
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
lasso.set_params(verbose=True)
print("Fitting Lasso Model")
print("-------------------")
print(lasso.fit(X_diabetes, y_diabetes))
print()
# The estimator automatically chose its lambda
print("Estimator alpha:", lasso.alpha_)