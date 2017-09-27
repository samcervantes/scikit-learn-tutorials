import matplotlib.pyplot as plt  
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

print(connectivity)