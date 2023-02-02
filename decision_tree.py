import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris


def load_dataset():
    """
    Loads iris dataset.
    :return: iris dataset
    """
    dataset = load_iris()

    return dataset


def describe_dataset(dataset):
    """
    Prints dataset description.
    :param dataset: dataset to describe.
    :return: None
    """
    print(f"Dataset feature names: {dataset.feature_names}")
    print(f"Dataset target names: {dataset.target_names}")


def prepare_data_and_target(dataset, to_remove):
    """
    Prepares data and target from the dataset by removing part of it.
    :param dataset: dataset to use.
    :param to_remove: list of indexes to remove from the dataset.
    :return: data and target
    """
    data = np.delete(dataset.data, to_remove, axis=0)
    target = np.delete(dataset.target, to_remove)
    return data, target


def train(data, target):
    """
    Creates and trains the decision tree classifier on the provided data and target.
    :param data: data to use for training.
    :param target: target to use for training.
    :return: trained decision tree classifier.
    """
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(data, target)
    return classifier


def predict(classifier, data, target):
    """
    Predict using the provided classifier and data.
    :param classifier: trained decision tree classifier.
    :param data: data to use for prediction.
    :param target: expected result.
    :return: prediction.
    """
    prediction = classifier.predict(data)
    print("Original Data:\n", data)
    print("Original Labels:", target)
    print("Labels Predicted:", prediction)
    return prediction


def draw(classifier, dataset):
    """
    Draws decision tree from the trained classifier.
    :param classifier: trained decision tree classifier.
    :param dataset: dataset used for training.
    :return: None
    """

    plt.figure(figsize=(50, 50))
    tree.plot_tree(classifier,
                   feature_names=dataset.feature_names,
                   class_names=dataset.target_names,
                   filled=True)
    plt.show()


if __name__ == "__main__":
    dataset = load_dataset()
    describe_dataset(dataset)
    data, target = prepare_data_and_target(dataset, [0, 50, 100])
    classifier = train(data, target)
    prediction = predict(classifier, dataset.data[[0, 50, 100]], dataset.target[[0, 50, 100]])
    draw(classifier, dataset)
