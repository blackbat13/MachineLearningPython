from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle


def prepare_image():
    image = load_sample_image("china.jpg")
    image = np.array(image, dtype=np.float64) / 255
    return image


def image_to_array(image):
    image = np.array(image, dtype=np.float64) / 255
    width, height, d = tuple(image.shape)
    image_array = np.reshape(image, (width * height, d))
    return width, height, image_array


def generate_kmeans(array, clusters_count):
    print("Fitting model on a small sub-sample of the data")
    start = time()

    array_sample = shuffle(array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=clusters_count, n_init="auto", random_state=0).fit(
        array_sample
    )

    stop = time()
    print(f"Done in {stop - start:0.3f}s.")

    return kmeans


def predict_labels(kmeans, array):
    print("Predicting labels (k-means)")
    start = time()

    labels = kmeans.predict(array)

    stop = time()
    print(f"Done in {stop- start:0.3f}s.")

    return labels


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)


def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


colors_count = 64
image = load_sample_image("china.jpg")
show_image(image)

width, height, image_array = image_to_array(image)

kmeans = generate_kmeans(image_array, colors_count)

labels = predict_labels(kmeans, image_array)

clustered_image = recreate_image(
    kmeans.cluster_centers_, labels, width, height)

show_image(clustered_image)

# plt.figure(1)
# plt.clf()
# plt.axis("off")
# # plt.title("Original image (96,615 colors)")
# plt.imshow(image)

# plt.figure(2)
# plt.clf()
# plt.axis("off")
# # plt.title(f"Quantized image ({colors_count} colors, K-Means)")
# plt.imshow(clustered_image)

# plt.show()
