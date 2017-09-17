from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

#iris = datasets.load_iris()
digits = datasets.load_digits()

print()
print("Dataset Contents:")
print("-----------------")
print(dir(digits))
print()
print("Dataset Description:")
print("--------------------")
print(digits.DESCR)
#print(digits.images)
print()
print("Dataset Data:")
print("-------------")
print(digits.data)
print()
print("Dataset Target:")
print("---------------")
print(digits.target)
print()
print("Dataset Target Names:")
print("---------------------")
print(digits.target_names)

# Create a new classifier instance, which has two methods: fit(X, y) and predict(T)
clf = svm.SVC(gamma=0.001, C=100.)

# Train the model (printing the output). Use all but the last value.
print()
print("Training the model:")
print("-------------------")
print(clf.fit(digits.data[:-1], digits.target[:-1]))

# Predict the last value of the data, printing the output
print()
print("Predicting the last value:")
print("--------------------------")
print(clf.predict(digits.data[-1:]))
print(digits.images[-1:])
plt.figure(1, figsize=(3,3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

