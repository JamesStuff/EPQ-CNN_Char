# spec_model.py of EPQ Project
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import adam_v2

TEST_RATIO, VALIDATE_RATIO = 0.2, 0.2


def gen_model(image_dim=(28, 28, 3)):
    num_of_filters = 60
    size_of_filter1 = (5, 5)
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    num_of_nodes = 500
    drop = 0.5

    # === A safe saved set of SAFE parameters ===
    # num_of_filters = 80
    # size_of_filter1 = (6, 6)
    # size_of_filter2 = (4, 4)
    # size_of_pool = (2, 2)
    # num_of_nodes = 600
    # drop = 0.5

    # num_of_filters = 40
    # size_of_filter1 = (4, 4)
    # size_of_filter2 = (2, 2)
    # size_of_pool = (2, 2)
    # num_of_nodes = 400
    # drop = 0.5

    # === Activation Functions ===
    # relu, softmax, tanh, sigmoid, softplus

    model = Sequential()
    model.add((Conv2D(num_of_filters, size_of_filter1, input_shape=(
        image_dim[0], image_dim[1], 1), activation='relu')))

    model.add((Conv2D(num_of_filters, size_of_filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(num_of_filters // 2, size_of_filter2, activation='relu')))
    model.add((Conv2D(num_of_filters // 2, size_of_filter2, activation='relu')))

    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(drop))

    model.add(Flatten())

    model.add(Dense(num_of_nodes, activation='relu'))
    model.add(Dropout(drop))

    model.add(Dense(10, activation='softmax'))

    model.compile(adam_v2.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
