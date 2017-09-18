"""
Scikit-learn requires an input dataset consisting of a 2D array of samples & features.
The Digits dataset consists of a 3D array (1797, 8, 8) so it must first be transformed (shaped)
into a 2D array.
"""

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
print("The shape of the digits dataset is:", digits.images.shape)

# Print the dataset desecription
print()
print(digits.DESCR)

# Plot the last image as an example
"""
import matplotlib.pyplot as plt 
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
plt.show()
"""

# Print the original 3D data array:
print()
print("Original 3D data:")
print("-----------------")
print(digits.images)

# Transform each 8x8 image into a feature vector of length 64
# This is the same as calling numpy.reshape(new-row-size, new-column-size)
# In this case we're converting the (1797, 8, 8) array into a (1797, 64) array so
#   we call digits.images.reshape(1797, -1) to preserve the 1797 rows and reduce 
#   the number of dimensions by 1
data = digits.images.reshape((digits.images.shape[0], -1))

# Print the new 2D data array:
print()
print("Reshaped 2D data:")
print("-----------------")
print(data)

# Now run classifier

# Create a new instance of a classifier object
clf = svm.SVC(gamma=0.001, C=100.)

# Train the model, using all but the last value
print()
print("Training the model:")
print("-------------------")
print(clf.fit(digits.data[:-1], digits.target[:-1]))

# Predict the last value using the model
print()
print("Based on the model, the prediction for the last value is:")
print("---------------------------------------------------------")
print(clf.predict(digits.data[-1:]))
print()







