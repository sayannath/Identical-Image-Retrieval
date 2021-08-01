"""
# Train KNN

## Initial-Setup
"""

# !nvidia-smi

# !cp -r '/content/drive/MyDrive/Similar-Image-Search/model_bit.h5' '/content/'

"""## Import the necessary modules"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications import *
import tensorflow_hub as hub

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Fix the random seeds
SEEDS=666

np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

"""## Data Gathering

Importing the Flower Dataset
"""

# Gather Flowers dataset
train_ds, validation_ds = tfds.load(
    "tf_flowers",
    split=["train[:85%]", "train[85%:]"],
    as_supervised=True,
)

"""#### Define the class"""

CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

"""#### Count of Training and Validation Samples"""

print("Number of Training Samples: ",len(train_ds))
print("Number of Validation Samples: ",len(validation_ds))

"""### Visualise the Dataset"""

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(CLASSES[int(label)])
    plt.axis("off")

"""## Define the Hyperparameters"""

IMAGE_SIZE = 160
BATCH_SIZE = 64
AUTO = tf.data.AUTOTUNE

"""## Dataloader"""

# Image preprocessing utils
def preprocess_test(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

"""## Creating the pipeline for validation sample"""

validation_ds = (
    validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTO)
)

"""## Load model into KerasLayer"""

model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
module = hub.KerasLayer(model_url, trainable=False)

"""## BiT Model"""

class MyBiTModel(tf.keras.Model):
  def __init__(self, module):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(128)
    self.normalize = Lambda(lambda a: tf.math.l2_normalize(a, axis=1))
    self.bit_model = module
  
  def call(self, images):
    bit_embedding = self.bit_model(images)
    dense1_representations = self.dense1(bit_embedding)
    return self.normalize(dense1_representations)

model = MyBiTModel(module=module)

"""## Load the weights of the trained BiT Model"""

model.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
model.load_weights("model_bit.h5")

"""### Checking the Validation Pipeline"""

images, labels = next(iter(validation_ds.take(1)))
print(images.shape, labels.shape)

random_index = int(np.random.choice(images.shape[0], 1))
plt.imshow(images[random_index])
plt.show()

"""## Train a Nearest Neighbors' Model

Determining out nearest neighbors for the features of our query image
"""

validation_features = model.predict(images)
start = time.time()
neighbors = NearestNeighbors(n_neighbors=5,
    algorithm='brute',
    metric='euclidean').fit(validation_features)
print('Time taken: {:.5f} secs'.format(time.time() - start))

"""### Determine the neighbors nearest to our query image"""

distances, indices = neighbors.kneighbors([validation_features[random_index]])
for i in range(5):
    print(distances[0][i])

"""### Visualize a neighbor"""

plt.imshow(images[indices[0][1]], interpolation='lanczos')
plt.show()

"""## Visualizing the nearest neighbors on images"""

def plot_images(images, labels, distances):
    plt.figure(figsize=(20, 10))
    columns = 4
    for (i, image) in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + "Label: {}".format(CLASSES[labels[i]]))
        else:
            ax.set_title("Similar Image # " + str(i) +
                         "\nDistance: " +
                         str(float("{0:.2f}".format(distances[i]))) + 
                         "\nLabel: {}".format(CLASSES[labels[i]]))
        plt.imshow(image)

for i in range(6):
    random_index = int(np.random.choice(images.shape[0], 1))
    distances, indices = neighbors.kneighbors(
        [validation_features[random_index]])
    
    # Don't take the first closest image as it will be the same image
    similar_images = [images[random_index]] + \
        [images[indices[0][i]] for i in range(1, 4)]
    similar_labels = [labels[random_index]] + \
        [labels[indices[0][i]] for i in range(1, 4)]
    plot_images(similar_images, similar_labels, distances[0])

"""## Visualizing the embedding space for the current validation batch"""

tsne_results = TSNE(n_components=2).fit_transform(validation_features)

color_map = plt.cm.get_cmap('coolwarm')
scatter_plot = plt.scatter(tsne_results[:, 0],
                           tsne_results[:, 1],
                           c=labels,
                           cmap=color_map)
plt.colorbar(scatter_plot)
plt.show()

"""## Visualizing the embedding space for the entire validation pipeline"""

validation_labels = [label
    for _, labels in validation_ds for label in labels
]
print(len(validation_labels))

validation_features = model.predict(validation_ds)

tsne_results = TSNE(n_components=2).fit_transform(validation_features)

color_map = plt.cm.get_cmap('coolwarm')
scatter_plot = plt.scatter(tsne_results[:, 0],
                           tsne_results[:, 1],
                           c=validation_labels,
                           cmap=color_map)
plt.colorbar(scatter_plot)
plt.show()