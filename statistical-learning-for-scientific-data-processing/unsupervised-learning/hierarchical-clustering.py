print(__doc__)

import time as time

import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
import scipy as sp
import numpy as np

# Generate data
try:
	face = sp.face(gray=True)
except AttributeError:
	from scipy import misc
	face = misc.face(gray=True)

print("face begins as a", face.shape, type(face))

# Resize it to 10% of the original size to speed up processing
face = misc.imresize(face, 0.10) / 255.
print("then we resize it to 10% of it's original size, giving it a new shape of", face.shape)

X = np.reshape(face, (-1,1))
print("next, we reshape this new array to a shape of", X.shape)

# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*face.shape)
print("Then we run grid_to_graph which returns a", type(connectivity), "with the shape", connectivity.shape)

# Compute clustering
st = time.time()
n_clusters = 15 # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity)
print()
print("Fitting the data to an AgglomerativeClustering model:")
print(ward.fit(X))
label = np.reshape(ward.labels_, face.shape)
print("Elapsed time:", time.time() - st)
print("Number of pixesl:", label.size)
#print("Number of clusters:", np.unique(label),size)

# Show the before & after images
fig = plt.figure()
a = fig.add_subplot(1,2,1)
imgplot = plt.imshow(face)
a.set_title("Raw Image")
a = fig.add_subplot(1,2,2)
imgplot = plt.imshow(face, cmap=plt.cm.gray)
for l in range(n_clusters):
	plt.contour(label == l, contours=1, colors=[plt.cm.spectral(l / float(n_clusters)), ])
a.set_title("Segmented Image")
plt.xticks(())
plt.yticks(())
plt.show()