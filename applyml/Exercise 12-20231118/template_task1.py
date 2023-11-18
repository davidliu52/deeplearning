# -*- coding: utf-8 -*-
"""
Template for Task 1 of Exercise 12

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de

"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from tensorflow import keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

(x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.cifar10.load_data()
assert x_train_raw.shape == (50000, 32, 32, 3)
assert x_test_raw.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# turn softmax probabilities into one-hot encoded array
def binarize(y):
    y_bin = np.zeros_like(y)
    for i, y_ in enumerate(y):
        idx_max = np.argmax(y_)
    
        y_bin[i, idx_max] = 1.0
    return y_bin

# one-hot encode labels
enc = OneHotEncoder(sparse_output=False)
enc.fit(y_train)
y_train_ohe = enc.transform(y_train)
y_test_ohe = enc.transform(y_test)

# show a single sample
plt.figure()
plt.imshow(x_train_raw[10])
plt.show()

# re-shape data into  [N, m] from [N, 32, 32, 3]
new_dim = x_train_raw.shape[1] * x_train_raw.shape[2] * x_train_raw.shape[3]
x_train = x_train_raw.reshape(x_train_raw.shape[0], new_dim)
x_test = x_test_raw.reshape(x_test_raw.shape[0], new_dim)

input_shape = x_train.shape[1]
output_shape = y_train_ohe.shape[1]


# build a neural network using TF's sequential API
