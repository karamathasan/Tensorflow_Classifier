import PIL.Image
import tensorflow as tf
import keras
from keras import Sequential 
from keras import layers
from keras import models

import PIL
import pathlib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
print("\n\n\n\n")

# we have a csv of image links and the species of butterfly in the image
# we can make a conv net to train on each image as we iterate through the csv, calulate error with the labels
# after using the labels, test on the other csv and then record the accuracy
# bonus is to use unsupervised learning
data = pd.read_csv("data/butterflyData/Training_set.csv")

img_height = 224
img_width = 224
num_classes = 75

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(\
    optimizer='adam',\
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),\
    metrics=[keras.metrics.CategoricalAccuracy()]\
    )

imgs = pathlib.Path("data/butterflyData")
train_imgs = list(imgs.glob('train/*'))
test_imgs = list(imgs.glob('test/*'))

# epochs = 10
# files = data['filename'][0:100]

# labelDataSet = data['label'][0:100]
one_hot_train = pd.get_dummies(data, columns=["label"], dtype=int)
one_hot_train = one_hot_train.drop("filename", axis=1)
one_hot_train = one_hot_train.iloc[0:100]

# print(one_hot.head())

for i in range(100):
    img = PIL.Image.open(train_imgs[i])
    img = keras.utils.img_to_array(img)
    y = one_hot_train.iloc[i].transpose()
    # print(f"img shape: {img.shape}")
    # print(f"classification shape: {y.shape}")
    # plt.imshow(keras.utils.img_to_array(img, dtype=int))
    # plt.title(data["label"].iloc[i])
    # plt.show()

    img = np.expand_dims(img, axis = 0)
    y = np.expand_dims(y, axis = 0)
    model.fit(img,y,batch_size=1)

testData = data = pd.read_csv("data/butterflyData/Testing_set.csv")

for i in range(10):
    img = PIL.Image.open(test_imgs[i])
    prediction = model.predict(np.expand_dims(keras.utils.img_to_array(img, dtype=float),axis=0))
    max_index = np.argmax(prediction)

    prediction = one_hot_train.columns[max_index]
    img = keras.utils.img_to_array(img, dtype=int)

    plt.imshow(img)
    plt.title(prediction)
    plt.show()
