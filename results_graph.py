# results_graph.py of EPQ Project
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

import matplotlib.pyplot as plt
import numpy as np

def history_graph(history=None):
    # Outputting of resluts from training, could be put into a live graph, but really
    # no point to be honest.

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title(['Training'])
    plt.xlabel('Epoch')

    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title(['Accuracy'])
    plt.xlabel('Epoch')

    plt.show()