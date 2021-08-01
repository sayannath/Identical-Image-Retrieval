"""
# Train BigTransfer on Flower Dataset

## Initial-Setup
"""

# !nvidia-smi

# !pip install -q tensorflow-addons

"""## Import the necessary modules"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications import *
import tensorflow_hub as hub
import tensorflow_addons as tfa

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

"""## Training setup"""

#@title Set dataset-dependent hyperparameters

IMAGE_SIZE = "=\u003C96x96 px" #@param ["=<96x96 px","> 96 x 96 px"]
DATASET_SIZE = "\u003C20k examples" #@param ["<20k examples", "20k-500k examples", ">500k examples"]

if IMAGE_SIZE == "=<96x96 px":
  RESIZE_TO = 160
  CROP_TO = 128
else:
  RESIZE_TO = 512
  CROP_TO = 480

if DATASET_SIZE == "<20k examples":
  SCHEDULE_LENGTH = 500
  SCHEDULE_BOUNDARIES = [200, 300, 400]
elif DATASET_SIZE == "20k-500k examples":
  SCHEDULE_LENGTH = 10000
  SCHEDULE_BOUNDARIES = [3000, 6000, 9000]
else:
  SCHEDULE_LENGTH = 20000
  SCHEDULE_BOUNDARIES = [6000, 12000, 18000]

"""## Define the Hyperparameters"""

BATCH_SIZE = 64
NUM_CLASSES = 5
SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE
STEPS_PER_EPOCH = 10
DATASET_NUM_TRAIN_EXAMPLES = len([image for image in train_ds])
AUTO = tf.data.AUTOTUNE
CSV_PATH = 'train_bit.csv'

"""## Dataloader Function"""

def preprocess_train(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [RESIZE_TO, RESIZE_TO])
    image = tf.image.random_crop(image, [CROP_TO, CROP_TO, 3])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_test(image, label):
    image = tf.image.resize(image, [RESIZE_TO, RESIZE_TO])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

"""## Create the Data Pipeline"""

pipeline_train = (
    train_ds.shuffle(10000)
    .repeat(
        int(SCHEDULE_LENGTH * BATCH_SIZE / DATASET_NUM_TRAIN_EXAMPLES * STEPS_PER_EPOCH)
        + 1
        + 50
    )
    .map(preprocess_train, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

pipeline_test = (
    validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

"""## Visualise the dataset"""

image_batch, label_batch = next(iter(pipeline_train))

plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n+1)
    plt.imshow(image_batch[n])
    plt.title(CLASSES[label_batch[n].numpy()])
    plt.axis('off')

"""## Load model into KerasLayer"""

model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
module = hub.KerasLayer(model_url, trainable=True)

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

"""## Define the optimiser and loss"""

lr = 0.003 * BATCH_SIZE / 512 

# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES, 
                                                                   values=[lr, lr*0.1, lr*0.001, lr*0.0001])

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

loss_fn = tfa.losses.TripletSemiHardLoss()

"""### Compile the Model"""

model.compile(optimizer=optimizer, loss=loss_fn)

"""## Setting up Callback"""

train_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode="auto", restore_best_weights=True),
    tf.keras.callbacks.CSVLogger(CSV_PATH),
]

"""## Plot the results"""

def plot_training(H, embedding_dim):
    with plt.xkcd():
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.title("Embedding dim: {}".format(embedding_dim))
        plt.legend(loc="lower left")
        plt.show()

"""## Train the `BiT` Model"""

print("Training started!")

start = time.time()
history = model.fit(
    pipeline_train,
    batch_size=BATCH_SIZE,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs= int(SCHEDULE_LENGTH / STEPS_PER_EPOCH),  
    validation_data=pipeline_test,
    callbacks=train_callbacks                                   
)

end = time.time()-start
print("Model takes {} seconds to train".format(end))

plot_training(history, 128)

"""## Save the `BiT` model"""

KERAS_FILE = "model_bit.h5"
model.save_weights(KERAS_FILE)

# !cp -r '/content/model_bit.h5' '/content/drive/MyDrive/Similar-Image-Search/'

# !cp -r '/content/train_bit.csv' '/content/drive/MyDrive/Similar-Image-Search/'

"""Saved the Model"""