# Compares the results of the knn and logistic regression models for classifying digits

from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# Set aside the last 10% of data (179 samples out of 1797 total) for testing
X_train = X_digits[:-179]
y_train = y_digits[:-179]
X_test = X_digits[179:]
y_test = y_digits[179:]

print("y_test values:", y_test)
print()

# First try a linear model - logistic regression
print("Logistic Regression")
print("-------------------")
logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)
logistic_predicted = logistic.predict(X_test)
print("Predicted values:", logistic_predicted)

# Next try K nearest neighbors model
print()
print("K Nearest Neighbors")
print("-------------------")
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
print("Preicted values:", knn_predicted)

print()
print("*** TEST DATA ***")
print("Actual Value    Predicted Value    Predicted Value    Result")
print("                  (Regression)          (knn)")
print("-------------------------------------------------------------")
for i in range(179):
	print("    ", y_test[i], "               ", logistic_predicted[i], "                ", knn_predicted[i], "         ", "Correct" if y_test[i] == logistic_predicted[i] and y_test[i] == knn_predicted[i] else "INCORRECT")

print()
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))