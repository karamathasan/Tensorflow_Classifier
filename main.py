import tensorflow as tf
import keras
from keras import Sequential 
from keras import layers
from keras import models

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import PIL
import PIL.Image
import pathlib

from helper import shuffle, dataSplit

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
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
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

dir = pathlib.Path("data/butterflyData")
imgs = list(dir.glob('train/*'))
# test_imgs = list(dir.glob('test/*'))

classification = pd.get_dummies(data, columns=["label"], dtype=int)
classification = classification.drop("filename", axis=1)

training_imgs, training_classes, testing_imgs, testing_classes = dataSplit(imgs, classification)

# /////// ITERATIVE FITTING ///////
for j in range(1):
    shuffle(training_imgs, training_classes)
    # for i in range(len(training_imgs)):
    for i in range(100):
        img = PIL.Image.open(training_imgs[i])
        img = keras.utils.img_to_array(img)
        y = training_classes.iloc[i].transpose()

        img = np.expand_dims(img, axis = 0)
        y = np.expand_dims(y, axis = 0)
        model.fit(img,y,batch_size=1)

score = 0
iterations = 100
for i in range(iterations):
    img = PIL.Image.open(testing_imgs[i])
    prediction = model.predict(np.expand_dims(keras.utils.img_to_array(img,dtype=float ),axis = 0))
    true_class = classification.columns[np.argmax(testing_classes.iloc[i])]
    predicted_class = classification.columns[np.argmax(prediction)]
    # print(f'    prediction: {prediction}')
    print()
    print(f'    true class: {true_class}, predicted class: {predicted_class}')
    if (testing_classes.iloc[i].to_numpy().all() == prediction.all()):
        score+=1
print(f"accuracy: {100 * (score / iterations)}%")

# for i in range(10):
#     img = PIL.Image.open(testing_imgs[i])
#     prediction = model.predict(np.expand_dims(keras.utils.img_to_array(img, dtype=float),axis=0))
#     max_index = np.argmax(prediction)

#     prediction = classification.columns[max_index]
#     img = keras.utils.img_to_array(img, dtype=int)

#     plt.imshow(img)
#     plt.title(prediction)
#     plt.show()
