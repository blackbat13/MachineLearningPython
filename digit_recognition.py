import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical


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
        pl.title(np.where(titles[i] == titles[i].max())[0][0], size=12)
        pl.xticks(())
        pl.yticks(())

    plt.show()


def prepare_labels(labels):
    """
    Prepares labels for the network.
    :param labels: labels to prepare.
    :return: prepared labels.
    """
    return to_categorical(labels)


def prepare_images(images, size):
    """
    Prepares images for the network.
    :param images: images to prepare.
    :param size
    :return: prepared images.
    """
    return (images.reshape((size, 28 * 28))).astype('float32') / 255


def create_network():
    """
    Creates neural network.
    :return: neural network.
    """
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network


def train_network(network, data, target):
    """
    Trains neural network.
    :param network: neural network.
    :param data: data for training.
    :param target: target for training data.
    :return: trained network.
    """
    network.fit(data, target, epochs=5, batch_size=128)
    return network


def evaluate(network, data, target):
    """
    Evaluates network on provided data.
    :param network: trained neural network to evaluate.
    :param data: data to use for the evaluation.
    :param target: expected results.
    :return: None
    """

    test_loss, test_acc = network.evaluate(data, target)
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)


def predict(network, data):
    """
    Use network for prediction.
    :param network: trained neural network.
    :param data: data for the prediction.
    :return: prediction.
    """
    return network.predict(data)


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_labels = prepare_labels(train_labels)
    test_labels = prepare_labels(test_labels)
    train_images = prepare_images(train_images, 60000)
    test_images = prepare_images(test_images, 10000)
    plot_gallery(test_images, test_labels, 28, 28)
    network = create_network()
    network = train_network(network, train_images, train_labels)
    evaluate(network, test_images, test_labels)
    prediction = predict(network, test_images)
    plot_gallery(test_images, prediction, 28, 28)
