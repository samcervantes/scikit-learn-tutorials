print(__doc__)

# From scikit-learn.org:
# ----------------------
# C Corresponds to regularize more the estimation
# Increasing C yeilds s more complex model (more feature are selected)

# From stackoverflow:
"""
The C parameter tells the SVM optimization how much you want to avoid misclassifying each training 
example. For large values of C, the optimization will choose a smaller-margin hyperplane if that 
hyperplane does a better job of getting all the training points classified correctly. Conversely, a 
very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, 
even if that hyperplane misclassifies more points. For very tiny values of C, you should get 
misclassified examples, often even if your training data is linearly separable.
"""


import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10,0,10)

scores = list()
scores_std = list()

for C in C_s:
	svc.C = C
	# Note: n_jobs is the numbers of CPU's to use to do the computation. -1 means all CPU's
	this_scores = cross_val_score(svc, X, y, n_jobs=1)
	scores.append(np.mean(this_scores))
	scores_std.append(np.std(this_scores))

import matplotlib.pyplot as plt 
plt.figure(1, figsize=(4,3))
plt.clf()
plt.semilogx(C_s, scores)
plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV Score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()
