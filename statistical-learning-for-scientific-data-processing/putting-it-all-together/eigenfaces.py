"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt 

import pandas
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC 

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Download the data, if not already on disk and load it as numpy arrays
print()
print("Importing the LFW People dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print(dir(lfw_people))
print("---", lfw_people.DESCR, "shapes ---")
print("Data:", lfw_people.data.shape)
print("Images:", lfw_people.images.shape)
print("target:", lfw_people.target.shape)
print("tagget_names:", lfw_people.target_names.shape)
print()

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1] # we have 50x37=1850 pixels (features)

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("--- Total dataset size ---")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print()

# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("--- Set aside 25% of the data for test ---")
print("X_train:", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)
print()

# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled dataset): unsupervised feature extraction / dimensionality reduction.
n_components = 150

print("Extracting the top %d eigenfaces from %d faces in X_train..." % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("...done in %0.3fs" % (time() - t0))
print()

print("pca.components_:", pca.components_.shape)
eigenfaces = pca.components_.reshape((n_components, h, w))
print("eigenfaces:", eigenfaces.shape)
print()

print("Projecting the input data on the eigenfaces orthonormal basis...")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("...done in %0.3fs" % (time() - t0))
print()
print("X_train_pca:", X_train_pca.shape)
print("X_test_pca:", X_test_pca.shape)
print()

# Train a SVM classification model
print("Fitting the classifier to the training set. This may take a while...")
t0 = time()
param_grid = { 'C': [ 1e3, 5e3, 1e4, 5e4, 1e5 ], 
	'gamma': [ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1 ], }
clf = GridSearchCV(SVC(kernel ='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("...done in %0.3fs" % (time() - t0))
print()
print(clf)
print()
print("Best estimator found by grid search:", clf.best_estimator_)
print()

# Quantitiative evaluation of the model quality on the test set
print("Predicting people's names on the test set...")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("...done in %0.3fs" % (time() - t0))
print()
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
	"""Helper function to plot a gallery of portraits"""
	plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
	plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.9, hspace=.35)
	for i in range(n_row * n_col):
		plt.subplot(n_row, n_col, i + 1)
		plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
		plt.title(titles[i], size=12)
		plt.xticks(())
		plt.yticks(())


# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
	pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
	true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
	return 'predicted: %s\ntrue:       %s' % (pred_name, true_name)


prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significant eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()






