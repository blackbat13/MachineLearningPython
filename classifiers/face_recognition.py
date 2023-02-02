import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def plot_gallery(images, titles, height, width, n_row=3, n_col=6):
    """
    Plots gallery of images and labels.
    :param images: array of images.
    :param titles: titles of the images.
    :param height: height of the image.
    :param width: width of the image.
    :param n_row: number of rows.
    :param n_col: number of columns.
    :return: None
    """
    pl.figure(figsize=(1.7 * n_col, 2.3 * n_row))
    pl.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.9, hspace=0.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((height, width)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

    plt.show()


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def plot_histogram(data):
    """
    Plot histogram of the provided data.
    :param data: data to use.
    :return: None
    """
    pl.figure(figsize=(14, 18))

    unique_data = np.unique(data)
    counts = [(data == i).sum() for i in unique_data]

    pl.xticks(unique_data, names[unique_data])
    locs, labels = pl.xticks()
    pl.setp(labels, rotation=45, size=20)
    _ = pl.bar(unique_data, counts)
    plt.show()


if __name__ == "__main__":
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    data = lfw_people.data
    target = lfw_people.target
    names = lfw_people.target_names

    samples_count, features_count = data.shape
    _, height, width = lfw_people.images.shape
    classes_count = len(names)

    print(f"Samples count: {samples_count}")
    print(f"Features count: {features_count}")
    print(f"Classes count: {classes_count}")

    plot_gallery(data, names[target], height, width)

    plot_histogram(target)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

    n_components = 150
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(x_train)

    eigenfaces = pca.components_.reshape((n_components, height, width))
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, height, width)

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    print("Fitting the classifier to the training set")
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier = classifier.fit(x_train_pca, y_train)

    print("Best estimator found by grid search:")
    print(classifier.best_estimator_)

    print("Predicting people's names on the test set")
    y_prediction = classifier.predict(x_test_pca)

    print(classification_report(y_test, y_prediction, target_names=names))
    print(confusion_matrix(y_test, y_prediction, labels=range(classes_count)))

    prediction_titles = [title(y_prediction, y_test, names, i)
                         for i in range(y_prediction.shape[0])]

    plot_gallery(x_test, prediction_titles, height, width)
