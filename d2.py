# d2.py of EPQ Project
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

import numpy as np
import pickle
import cv2

from time import time, sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WIDTH = 1920
HEIGHT = 1080

NORMALISE = False

cap = cv2.VideoCapture(1)
cap.set(3, WIDTH)
cap.set(3, HEIGHT)

pf = open("models/model_train2.pickle", 'rb')
model = pickle.load(pf)

print(model.summary())


def pre_proc(img, normalise=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if normalise:
        img = cv2.equalizeHist(img)

    img = img / 255

    return img


st = time()
t = time()
p2 = {}

print("\n" * 5)

re_img = None
re_img_original = None


def update_animation(i):
    success, img_original = cap.read()

    # Properly crop image...
    center = img_original.shape[0] / 2, img_original.shape[1] / 2
    x, y = center[1] - 1080 / 2, center[0] - 1080 / 2

    img_original = img_original[int(y):int(y + 1080), int(x):int(x + 1080)]

    img = np.asarray(img_original)
    img = cv2.resize(img, (28, 28))
    img = pre_proc(img)

    re_img = cv2.resize(img, (420, 420), interpolation=cv2.INTER_NEAREST)
    re_img_original = cv2.resize(img_original, (420, 420), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Pre-Processed Image (CNN Input)", re_img)
    cv2.imshow("Webcam Input", re_img_original)

    img = img.reshape(1, 28, 28, 1)
    pred = model.predict(img)

    class_index = int(np.argmax(pred, axis=1))

    for index, i in enumerate(pred[0]):
        p2[index] = int(i * 10000) / 100

    t = time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit()

    # ==== Animation Function Stuff =====
    x_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_vals = pred[0]

    plt.cla()

    y_pos = np.arange(len(y_vals))
    plt.ylim(top=1)
    plt.bar(y_pos, y_vals)
    plt.xticks(y_pos, x_vals)


# with plt.xkcd():
# with None:
ani = FuncAnimation(plt.gcf(), update_animation, interval=40)

plt.tight_layout()
plt.show()
