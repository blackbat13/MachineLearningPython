import time as time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.data import coins
from skimage.transform import rescale
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph


def show_image(image, clusters_count=0, label=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=plt.cm.gray)
    for i in range(clusters_count):
        plt.contour(
            label == i,
            colors=[
                plt.cm.nipy_spectral(i / float(clusters_count)),
            ],
        )

    plt.axis("off")
    plt.show()


def prepare_coins():
    original_coins = coins()

    smoothened_coins = gaussian_filter(original_coins, sigma=2)

    rescaled_coins = rescale(
        smoothened_coins,
        0.2,
        mode="reflect",
        anti_aliasing=False,
    )

    return rescaled_coins


def compute(x, connectivity, image, n_clusters):
    print("Computing structured hierarchical clustering...")

    start = time.time()

    ward = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward", connectivity=connectivity
    )
    ward.fit(x)

    label = np.reshape(ward.labels_, prepared_coins_image.shape)

    stop = time.time()

    print(f"Elapsed time: {stop - start:.3f}s")
    print(f"Number of pixels: {label.size}")
    print(f"Number of clusters: {np.unique(label).size}")

    return label


show_image(coins())

prepared_coins_image = prepare_coins()

x = np.reshape(prepared_coins_image, (-1, 1))

connectivity = grid_to_graph(*prepared_coins_image.shape)

clusters_count = 27

label = compute(x, connectivity, prepared_coins_image, clusters_count)

show_image(prepared_coins_image, clusters_count, label)
