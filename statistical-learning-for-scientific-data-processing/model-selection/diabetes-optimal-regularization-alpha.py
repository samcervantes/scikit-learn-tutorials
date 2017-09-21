from __future__ import print_function
print(__doc__)


from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.data[:150]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 3

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.figure()
plt.semilogx(alphas, scores)

# plot error lines showsing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])
plt.show()

# How much can we trust this selection of alpha?
#
# To answer this question we use the LassoCV object that sets its alpha
# parameter automatically from the data by internal cross-validation (i.e. it
# performs cross validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.
lasso_cv = LassoCV(alphas = alphas, random_state = 0)
k_fold = KFold(3)

print("How much can you trust the selection of alpha?")
print("Alpha parameters maximizing the generalization score on different subsets of the data:")
for k, (train, test) in enumerate(k_fold.split(X, y)):
	lasso_cv.fit(X[train], y[train])
	print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
		format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()
print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores of these alphas differ")
print("quite substantially.")

plt.show()




