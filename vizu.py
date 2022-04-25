# vizu.py of EPQ Project FINAL NAME HERE
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

from keras.utils.vis_utils import plot_model
import cornichon

if __name__ == "__main__":
    model = cornichon.load("models/model_train2.pickle")
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)