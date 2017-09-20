# Grid search uses three-fold cross validation and an array of C values and chooses pararmeters
#   to maximize the cross validation score

from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np 
from sklearn import datasets, svm

digits = datasets.load_digits()
X_digits = digits.data
y_digits= digits.target

Cs = np.logspace(-6, -1, 10)
svc = svm.SVC(kernel='linear')
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)

clf.fit(X_digits[:1000], y_digits[:1000])

print("Best cross-validation score (on training data):", clf.best_score_)
print("Best estimator C:", clf.best_estimator_.C)

# Prediction performance on test set is not as good as on train set
print("Prediction score on test set:", clf.score(X_digits[1000:], y_digits[1000:]))