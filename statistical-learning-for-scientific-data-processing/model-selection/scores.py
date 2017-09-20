from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

svc = svm.SVC(C=1, kernel='linear')

score = svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])

print("Scoring by setting aside the last 100 samples for training data")
print("---------------------------------------------------------------")
print("Score:", score)
print()

import numpy as np 
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)

scores = list()

for k in range(3):
	# Use 'list' to copy in order to 'pop' later on
	X_train = list(X_folds)
	
	# Set X_test equal to the kth value in the X_folds training data, in the process removing that data from X_train
	X_test = X_train.pop(k)
	
	# Concatenate the remaining training data into a single numpy array
	X_train = np.concatenate(X_train)

	# Repeat the same procedure for the target data
	y_train = list(y_folds)
	y_test = y_train.pop(k)
	y_train = np.concatenate(y_train)
	scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

print("Scoring with hand-rolled Kfold cross validation (k=3)")
print("-----------------------------------------------------")
print("Scores:", scores)
print()

k_fold = KFold(n_splits=3)
scores = cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)

print("Scoring with KFold cross_val_score()")
print("------------------------------------")
print("Scores:", scores)





