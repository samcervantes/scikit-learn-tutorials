# A scikit learn array is a 2D array. The first axis is samples, the second axis is features:
# 				Feature 1	Feature 2	Feature 3	...
# 			---------------------------------------------
# Sample 1	|
# Sample 2	|
# Sample 3	|
# ...		|

# Load up the built-in datasets
from sklearn import datasets

# create a new instance of the classic iris dataset
iris = datasets.load_iris()

# get the 'data' attribute of the iris object - this has the features/samples array
data = iris.data

print("The Iris Dataset")
print("----------------")
print()
print("The shape of the data is:")
print(data.shape)
print()
print("This means the data array has", data.shape[0], "samples (obsercations) and", data.shape[1], "features")
print()
print("Dataset features: ", iris.feature_names)
print()
print("Raw Data:")
print(data)
