# testing.py of EPQ Project FINAL NAME HERE
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

import numpy as np
import pickle
import cv2

from time import time
import matplotlib.pyplot as plt

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
    # center = img.shape[0] / 2, img.shape[1] / 2
    # x = center[1] - 420 / 2
    # y = center[0] - 420 / 2
    #
    # img = img[int(y):int(y + 420), int(x):int(x + 420)]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if normalise:
        img = cv2.equalizeHist(img)

    img = img / 255

    return img

# Func to be removed
def bar_chart(pred=None):
    print("REMOVE DEPRECIATED FUNCTION IN TESTING.py\n")
    if pred is None:
        pred = []


# ========== Plotting ==========
plt.ion()



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

x = [*range(10)]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

plt.title("Predictions from Netowrk", fontsize=18)

plt.xlabel("Number", fontsize=18)
plt.ylabel("Confidence", fontsize=18)

# ========== Main bit ==========

st = time()
t = time()
p2 = {}

re_img = None
re_img_original = None
while True:
    success, img_original = cap.read()
    # Properly crop image...
    center = img_original.shape[0] / 2, img_original.shape[1] / 2
    x, y = center[1] - 1080 / 2, center[0] - 1080 / 2

    img_original = img_original[int(y):int(y + 1080), int(x):int(x + 1080)]

    img = np.asarray(img_original)
    # img = cv2.resize(img, (420, 420))
    img = cv2.resize(img, (28, 28))
    img = pre_proc(img)

    re_img = cv2.resize(img, (420, 420), interpolation=cv2.INTER_NEAREST)
    re_img_original = cv2.resize(img_original, (420, 420), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Pre-Processed Image (CNN Input)", re_img)
    cv2.imshow("Webcam Input", re_img_original)
    #
    # cv2.imwrite("img.png", re_img)
    # cv2.imwrite("img_original.png", re_img_original)

    # cv2.imshow("Combined", cv2.hconcat([re_img, re_img_original]))

    img = img.reshape(1, 28, 28, 1)
    pred = model.predict(img)

    # bar_chart(pred=pred[0])

    class_index = int(np.argmax(pred, axis=1))

    for index, i in enumerate(pred[0]):
        p2[index] = int(i * 10000) / 100
    print(" " * 150, end="\r")

    # ========= Choices ==========
    # print(" " * 150, end="\r")
    print(" " * 5, end="\r")

    # print(class_index, "-", f"{p2[class_index]}%  - {1 / (time() - t)}c/s -", p2, end="\r")
    print(class_index, end="\r")


    # # Plotting...
    # y = pred[0]
    #
    # ax.bar(x, y)
    # fig.canvas.draw()
    # fig.canvas.flush_events()

    t = time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
