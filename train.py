# train.py of EPQ Project
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

# Sample data contains 50,000 images for training data
# 10,000 for val

import os
import cv2
import pickle

from time import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

# imports left in to please pycharm and autocompletion if needed
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import adam_v2

from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import numpy as np

from spec_model import gen_model
from results_graph import history_graph

PATH = 'data'
TEST_RATIO, VALIDATE_RATIO = 0.2, 0.2
IMAGE_DIM = (28, 28, 3)

BATCH_SIZE = 50
EPOCHS = 6
STEPS_PER_EPOCH = 3000

dir_list = os.listdir(PATH)
print(f"dir_list:\t{dir_list}\nlen(...):\t{len(dir_list)}")  # Number of classes for len(...)

imgs = []
class_number = []

num_of_classes = 10

for i in range(0, len(dir_list)):
    pic_list = os.listdir(f"{PATH}/{i}")

    for j in pic_list:
        current_image = cv2.imread(f"{PATH}/{i}/{j}")

        current_image = cv2.resize(current_image, (IMAGE_DIM[0], IMAGE_DIM[1]))

        # Repetition of 3 blocks here is for duplication of the training data
        # Seems to have a net positive effect on the training.
        imgs.append(current_image)
        imgs.append(current_image)
        imgs.append(current_image)

        class_number.append(i)
        class_number.append(i)
        class_number.append(i)

    print(i, end=" -> ")

print("lens...", len(imgs), len(class_number))

imgs = np.array(imgs)
class_number = np.array(class_number)

print("Showing shape and dimension of lists...")
print(imgs.shape)
print(class_number.shape, "\n" * 2)

# Using 80% train 20% test => 4:1 // (1/5)
X_train, X_test, y_train, y_test = train_test_split(imgs, class_number, test_size=TEST_RATIO)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VALIDATE_RATIO)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

# Fairness validation...
# Doesn't 'really' matter for this case
sample_nums = []
for i in range(10):
    print(f"{i} occurrences: {len(np.where(y_train == i)[0])}")
    sample_nums.append(len(np.where(y_train == i)[0]))

plt.figure(figsize=(7, 5))
plt.bar(range(10), sample_nums)
plt.title("Num of Images per Classification")
plt.xlabel("Classification")
plt.ylabel("Num of Images")

plt.show()


def pre_proc(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255

    return img


X_train = np.array(list(map(pre_proc, X_train)))
X_test = np.array(list(map(pre_proc, X_test)))
X_validation = np.array(list(map(pre_proc, X_validation)))

print("\n")

# Adding depth for Convolutional Nerual Network to work properly
print(X_train.shape, X_test.shape, X_validation.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
print(X_train.shape, X_test.shape, X_validation.shape)

# Augmentation of data...
data_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=8
)

data_gen.fit(X_train)

# Encoding
# TODO: Fix...
y_train = to_categorical(y_train, num_of_classes)
y_test = to_categorical(y_test, num_of_classes)
y_validation = to_categorical(y_validation, num_of_classes)

model = gen_model()

print(model.summary())

# Training of the model
history = model.fit_generator(
    data_gen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=(X_validation, y_validation),
    shuffle=1)

history_graph(history)

score = model.evaluate(X_test, y_test, verbose=0)

print(f"score\t{score[0]}\naccuracy\t{score[1]}")

# Saves model as pickle file
print("Pickling File...")
try:
    d = f'models/{int(time()) - 1000000000}_more_drop.pickle'
    pf = open(d, 'wb')
    pickle.dump(model, pf)
    pf.close()

    print(f"Saved to {d}")

except:
    # Open except being ok, as saving the model really isn't too complex
    # as it's just a pickled object...
    print("failed to save model... Will hang on an input...")
    input("Press [Enter] to continue...")
