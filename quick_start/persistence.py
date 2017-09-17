
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()

iris = datasets.load_iris()

X, y = iris.data, iris.target

print('Fitting model:')
print('--------------')
# Fit the model, printing the output
print(clf.fit(X,y))

# Persistence with Pickle
"""
import pickle

# Save the model
s = pickle.dumps(clf)

# Load the saved model
clf2 = pickle.loads(s)
"""

# Persistence with joblib.dump
from sklearn.externals import joblib

# Save the model to disk
joblib.dump(clf, 'filename.pkl')

# Load the saved model from disk
clf2 = joblib.load('filename.pkl')

# Predict the value of the first datapoint
print()
print('The prediction of the last value, based on the saved model is:')
print('--------------------------------------------------------------')
print(clf2.predict(X[0:1]))
