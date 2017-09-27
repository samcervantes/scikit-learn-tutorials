import numpy as np
from sklearn import decomposition

# Create a signal with only 2 useful dimensions
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]

print("Signal:")
print(X)

pca = decomposition.PCA()
print("Fitting the PCA model:")
print(pca.fit(X))
print()

print("Explained variance:")
print(pca.explained_variance_)

pca.n_components = 2
X_reduced = pca.fit_transform(X)

print()
print("X_reduced:")
print(X_reduced)
print("The shape of the reduced data is:", X_reduced.shape)

