import numpy as np 
from sklearn.svm import SVC

rng = np.random.RandomState(0)
# Create a 100x10 array and populate it with random smaple from a uniform distribution over [0,1)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5,10)

clf = SVC()
print(clf.set_params(kernel='linear').fit(X,y))
print(clf.predict(X_test))

print(clf.set_params(kernel='rbf').fit(X,y))
print(clf.predict(X_test))
