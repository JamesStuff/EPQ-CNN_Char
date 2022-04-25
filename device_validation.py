# device_validation.py of EPQ Project
# Copyright James Robinson 2021-2022
# All rights reserved
#
# James Robinson (8116) - The Burgate School and Sixth Form (58815)

import tensorflow as tf

# Just lists some useful information when running on other devices, and going from
# different hardware. Such as M1 MAX GPU (Metal) vs NVIDIA RTX Card (Tensor).
print(f"TensorFlow version: {tf.__version__}")
print(f"Found devices:\n{tf.config.list_physical_devices()}")
