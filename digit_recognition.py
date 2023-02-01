import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical


def plot_gallery(images, titles, h, w, n_row=3, n_col=6):
    pl.figure(figsize=(1.7 * n_col, 2.3 * n_row))
    pl.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.9, hspace=0.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(np.where(titles[i] == titles[i].max())[0][0], size=12)
        pl.xticks(())
        pl.yticks(())


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

plot_gallery(test_images, test_labels, 28, 28)
plt.show()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

result = network.predict(test_images)
plot_gallery(test_images, result, 28, 28)
plt.show()
