import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()

print(iris.feature_names)
print(iris.target_names)

# Define indexes of elements to remove
removed = [0, 50, 100]

# Prepare new target for learning phase
new_target = np.delete(iris.target, removed)

# Prepare new data for learning phase
new_data = np.delete(iris.data, removed, axis=0)

# Define decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train classifier on new data and new target
clf = clf.fit(new_data, new_target)

# Assign removed data as input for prediction
prediction = clf.predict(iris.data[removed])

# Print removed data
print("Original Data", iris.data[removed])

# Print target for the removed data
print("Original Labels", iris.target[removed])

# Print prediction for predicted data
print("Labels Predicted", prediction)

# Draw decision tree from the trained classifier
tree.plot_tree(clf,
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True)

# Show the decision tree
plt.show()
