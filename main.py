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

from sklearn.model_selection import train_test_split

print("\n\n\n\n")

# we have a csv of image links and the species of butterfly in the image
# we can make a conv net to train on each image as we iterate through the csv, calulate error with the labels
# after using the labels, test on the other csv and then record the accuracy
# bonus is to use unsupervised learning

img_height = 224
img_width = 224
num_classes = 75

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(\
    optimizer='adam',\
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),\
    metrics=[keras.metrics.CategoricalAccuracy()]\
    )

dir = "data/butterflyData/train/"
data = pd.read_csv("data/butterflyData/Training_set.csv")
files = data['filename']
labels = pd.get_dummies(data, columns=["label"], dtype=int)
labels = labels.drop("filename", axis=1)

X_train, X_test, y_train, y_test  = train_test_split(files, labels, shuffle = False)

epochs = 10
batch_size = 64

# /////// ITERATIVE FITTING ///////
for j in range(epochs):
    print(f"    epoch {j+1}/{epochs}")
    # shuffle(training_imgs, training_classes)
    X = []
    y = []
    for i in range(len(X_train)):
        if (i % batch_size == 0 or i == len(X_train)-1) and i != 0: 
            X = np.array(X)
            y = np.array(y)
            model.fit(X,y, batch_size=batch_size)
            X = []
            y = []
        img = PIL.Image.open(dir + X_train.iloc[i])
        img = keras.utils.img_to_array(img)
        X.append(img)
        y.append(y_train.iloc[i])

score = 0
iterations = 64
for i in range(iterations):
    img = PIL.Image.open(dir + X_test.iloc[i])
    prediction = model.predict(np.expand_dims(keras.utils.img_to_array(img,dtype=float ),axis = 0))
    true_class = labels.columns[np.argmax(y_test.iloc[i])]
    predicted_class = labels.columns[np.argmax(prediction)]
    print(f"    location: {X_test.iloc[i]}")
    print(f'    true class: {true_class}, predicted class: {predicted_class}')
    # if (y_test.iloc[i].to_numpy().all() == prediction.all()):
    if (true_class == predicted_class):
        score+=1
print(f"accuracy: {100 * (score / iterations)}%")

