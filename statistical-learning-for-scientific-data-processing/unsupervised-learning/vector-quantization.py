import scipy as sp
from sklearn import cluster
import numpy as np

# For images
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

try:
	face = sp.face(gray=True)
except AttributeError:
	from scipy import misc
	face = misc.face(gray=True)

print("Face is a", type(face) , "of dimension", face.shape)
print("--- Face Array ---")
print(face)
print()

X = face.reshape((-1,1))

print("So we reshape it into a", X.shape, "array that the model requires:")
print("--- X Array ---")
print(X)
print()
print("And then we feed X into the kmeans clustering model:")
k_means = cluster.KMeans(n_clusters=5, n_init=1)
print(k_means.fit(X))
print()
# numpy.squeeze() removes single-dimensional entries from the shape of an array
values = k_means.cluster_centers_.squeeze()
print("Values (cluster centers):", values)
print()
labels = k_means.labels_
print("Labels: ", labels)
print()
print("Then we build a new array for the compressed image using with the new values as pixel intensities")
print("--- Compressed Image ---")
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape
print("face_compressed:", face_compressed)
print("Shape:", face_compressed.shape)
print()

# Show the before & after images
fig = plt.figure()
a = fig.add_subplot(1,2,1)
imgplot = plt.imshow(face)
a.set_title("Before")
a = fig.add_subplot(1,2,2)
imgplot = plt.imshow(face_compressed)
a.set_title("After")
plt.show()

