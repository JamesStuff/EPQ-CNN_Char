# rnd.py of EPQ Project
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

import cv2
import numpy as np
import cornichon

if __name__ == "__main__":
    def pre_proc(img, normalise=False):
        if normalise:
            img = cv2.equalizeHist(img)

        img = img / 255

        return img

    # Path of image to be replaced with what ever you need converting
    img = cv2.imread("/Users/jimbo/Documents/LaTeX/EPQ Final/Figures/frog.png")

    img = np.asarray(img)
    img = cv2.resize(img, (28, 28))
    img = pre_proc(img)

    cv2.imwrite("/Users/jimbo/Documents/LaTeX/EPQ Final/Figures/frogpix.png", img)

    cv2.imshow("Image", cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_NEAREST))

    cv2.waitKey(0)
    cv2.destroyAllWindows()