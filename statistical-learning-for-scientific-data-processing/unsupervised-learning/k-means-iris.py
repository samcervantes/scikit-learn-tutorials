from sklearn import cluster, datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

iris = datasets.load_iris()
X_iris = iris.data 
y_iris = iris.target

k_means = cluster.KMeans(n_clusters=3)
print("Fitting the k-means model with the input data only:")
print(k_means.fit(X_iris))
print()
print("Methods available on the k_means model:", dir(k_means))
print()
print("K-Means Labels (every 10th data point):  ", k_means.labels_[::10])
print()
print("Predicted values (every 10th data point):", y_iris[::10])

